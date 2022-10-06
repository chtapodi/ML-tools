import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms
from torch import nn
import torch
from torch import optim
import torchvision.utils
# import matplotlib.pyplot as plt
from ASAM.asam import ASAM, SAM

import importlib
from torch.utils.tensorboard import SummaryWriter

# dlr = importlib.import_module("Discriminative-learning-rates-PyTorch.discriminativeLR")



# from Discriminative-learning-rates-PyTorch import discriminativeLR as dlr

class custom_warmup :
    def __init__(self, optimizer, num_warmup, lr=.01, lrf=.1, warmup_bias_lr=.1, momentum=0.937, warmup_momentum=0.8) :
        self.optimizer=optimizer
        self.num_warmup=num_warmup
        self.warmup_bias_lr=warmup_bias_lr
        self.momentum=momentum
        self.warmup_momentum=warmup_momentum
        self.last_lr=self.warmup_bias_lr
        self.batches=0
        self.lr=lr
        self.lf=lambda x: (1 - x / self.total_epochs) * (1.0 - lrf) + lrf  # linear

    def step(self, epoch=0) :
        self.batches+=1
        xi=[0, self.num_warmup]
        # accumulate = max(1, np.interp(ni, xi, [1, nbs / batch_size]).round())
        bias_lr=np.interp(self.batches, xi, [self.warmup_bias_lr, self.lr])
        lr=np.interp(self.batches, xi, [0.0, self.lr])
        self.last_lr=lr
        momentum=np.interp(self.batches, xi, [self.warmup_momentum, self.momentum])

        for j, x in enumerate(self.optimizer.param_groups):
            # bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
            #TODO look into param groups
            # if j==0 :
            #     x['lr']=bias_lr
            #     print("bias_lr", bias_lr)
            #
            # else :
            #     x['lr'] = lr
            #     print("LR", lr)
            x['lr'] = lr

            if 'momentum' in x:
                x['momentum'] = momentum

    def get_last_lr(self) :
        return self.last_lr




class custom_reduce_on_plateau :
    def __init__(self, optimizer, reduction_multiplier=0.333, patience=15, maximize=False, min_lr=1e-10, score_check=None) :
        self.optimizer=optimizer
        self.reduction_multiplier=reduction_multiplier
        self.steps=0
        self.maximize=maximize
        self.patience=patience

        self.min_lr=min_lr

        self.best_score=1000.0
        if self.maximize :
            self.best_score=0.0

        if score_check is not None :
            self.score_check=score_check


        self.last_lr=self.optimizer.param_groups[0]['lr']*self.reduction_multiplier

    def step(self, epoch=0,score=None) :
        if score is None :
            score=self.score_check()
        print(score)
        improved=False

        if (self.maximize and score>self.best_score) :
            improved=True
        elif (not self.maximize and score<self.best_score) :
            improved=True

        if improved :
            self.best_score=score
        else :
            self.steps+=1

        if self.steps > self.patience and self.last_lr>=self.min_lr:
            self.steps=0
            self.last_lr=self.optimizer.param_groups[0]['lr']*self.reduction_multiplier
            print("Reducing to", self.last_lr)
            for x in self.optimizer.param_groups:
                x['lr']=self.last_lr


    def get_last_lr(self) :
        return self.last_lr

class trainer :
    def __init__(self, model, train_dataset, val_dataset=None,
        loader_kwargs=None, loader_workers=4, batch_size=16, shuffle_loader=True,
        forward_function=None, backward_function=None, clip_grad_norm=True,
        val_function=None, report_function=None, criterion=None,
        validate_accuracy=False, validate=True,
        use_early_stopping=True, early_stopping_function=None, early_stopping_patience=15,
        optimizer='SGD', warmup_scheduler=None, warmup_epochs=5, accumulate_batches=1,
        use_discriminativeLR=False, discriminativeLR=.0001, forward_model=None,
        scheduler_names=[], scheduler_steps=[], scheduler_kwargs=None,
        SAM=False, ASAM=False, SWA=False, SWALR_epochs=5, SWA_lr=.05,
        lr=.01, max_lr=None, decay=0.0005, momentum=0.937, scheduler_period=10,
        device=None, checkpoint_dir="training_checkpoints/") :

        self.checkpoint_dir=checkpoint_dir

        self.device=device
        if self.device is None :
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        print("{} selected".format(self.device))

        # Scalers
        self.lr=lr
        self.momentum=momentum
        self.decay=decay

        self.max_lr=max_lr
        if self.max_lr is None :
            self.max_lr=self.lr*10

        self.SAM=SAM

        self.ASAM=ASAM
        if self.ASAM and self.SAM :
            printf("SAM and ASAM cannot be enabled at the same time")
            exit()
        self.minimizer=None

        self.model=model
        self.model.to(self.device)

        self.criterion=criterion
        if self.criterion is None :
            self.criterion=torch.nn.MSELoss()

        # if discriminativeLR :
        #     params, lr_arr, _ = dlr.discriminative_lr_params(self.model, slice(min_lr, self.lr)) #slice(min_lr,max_lr)
        #     for p in params :
        #         p['lr']=(float)(p['lr'])

        self.last_score=0.0

        self.loader_kwargs=loader_kwargs
        self.loader_workers=loader_workers
        self.train_dataset=train_dataset
        self.val_dataset=val_dataset
        self.batch_size=batch_size
        self.shuffle_loader=shuffle_loader
        self.clip_grad_norm=clip_grad_norm

        self.batches_per_epoch=int(len(self.train_dataset)/self.batch_size)
        self.accumulate_batches=accumulate_batches

        self.SWA=SWA
        self.SWALR_epochs=SWALR_epochs
        self.SWA_lr=SWA_lr
        if self.SWA :
            self.SWA_model = optim.swa_utils.AveragedModel(self.model)

        self.SWA_active=False #For training loop to quickly check if SWA model has been updated

        self.optimizer=optimizer
        if isinstance(self.optimizer, str): #If optimizer name passed in, generate optimizer
            self.optimizer=self.generate_optimizer(self.optimizer, momentum)


        self.warmup_scheduler=warmup_scheduler
        self.warmup_epochs=warmup_epochs
        self.warmup_batches=self.warmup_epochs*self.batches_per_epoch
        print(self.warmup_batches)
        if self.warmup_scheduler is None :
            self.warmup_scheduler=custom_warmup

        if scheduler_kwargs is None :
            scheduler_kwargs=[None for i in range(len(scheduler_names))]

        if len(scheduler_steps)!=len(scheduler_names) or len(scheduler_kwargs)!=len(scheduler_names):
            print("Number of schedulers and number of scheduler durations do not match")
            exit()
        self.scheduler_list=[]
        self.scheduler_checkpoints=[]
        self.scheduler_period=scheduler_period #for cosine schedulers
        self.scheduler=self.generate_scheduler(scheduler_names, scheduler_steps, scheduler_kwargs)


        self.forward_model =forward_model
        if self.forward_model is None :
            self.forward_model=self.standard_forward

        self.backward=backward_function
        if self.backward is None :
            self.backward=self.standard_backward

        self.validate=validate #In case validation is not wanted
        self.validate_accuracy=validate_accuracy
        self.val_function=val_function
        if self.val_function is None :
            self.val_function=self.standard_validation


        self.use_early_stopping=use_early_stopping
        self.early_stopping_patience=early_stopping_patience
        self.early_stopping_function=early_stopping_function
        if self.validate_accuracy :
            self.best_score=0.0
        else :
            self.best_score=100.0
        if self.early_stopping_function is None :
            self.early_stopping_function=self.standard_early_stopping


        self.last_epoch=0

        self.end_of_epoch_function=self.standard_function_end

        self.writer = SummaryWriter()


        self.num_steps=1

    def generate_optimizer(self, optimizer_name, momentum) :

        params=self.model.parameters()
        lr=self.lr
        optimizer=None
        optimizer_constructor=None

        if "Adam" in optimizer_name :
            if optimizer_name == "Adam" :
                optimizer_constructor=optim.Adam

            elif optimizer_name == "AdamW" :
                optimizer_constructor=optim.AdamW

            optimizer=optimizer_constructor(params, lr=lr, betas=(momentum, 0.999), weight_decay=self.decay)


        else :

            if optimizer_name == "RMSProp" :
                optimizer_constructor=optim.RMSProp
                optimizer=optimizer_constructor(params, lr=lr, momentum=momentum, weight_decay=self.decay)

            else :
                optimizer_constructor=optim.SGD #Default option
                optimizer=optimizer_constructor(params, lr=lr, momentum=momentum, nesterov=True, weight_decay=self.decay)

        if self.SAM :
            self.minimizer=SAM( optimizer, self.model)
        elif self.ASAM :
            self.minimizer=ASAM( optimizer, self.model)

        # self.optimizer=optimizer
        return optimizer


    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform(m.weight)
            m.bias.data.fill_(0.01)

    def get_last_score(self) :
        print("lasct score", self.last_score)
        return self.last_score

    def generate_scheduler(self, scheduler_names, scheduler_steps, kwarg_list) :
        # https://www.kaggle.com/code/isbhargav/guide-to-pytorch-learning-rate-scheduling/notebook
        # Scheduler checkpoints is the last step the associated scheduler should run at
        schedulers=[]
        self.scheduler_checkpoints=[]

        def create_scheduler(scheduler_name, num_steps, kwargs=None) :
            # TODO: update specific parameters, make them configurable
            for i in range(num_steps) :
                self.scheduler_list.append(scheduler_name)

            if scheduler_name == "warmup" :
                if kwargs is None :
                    return self.warmup_scheduler(self.optimizer, num_steps)
                else :
                    return self.warmup_scheduler(self.optimizer, **kwargs)

            elif scheduler_name == "CosineAnnealingLR" :
                if kwargs is None :
                    return optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.scheduler_period)
                else :
                    return optim.lr_scheduler.CosineAnnealingLR(self.optimizer, **kwargs)

            elif scheduler_name == "CosineAnnealingWarmRestarts" :
                if kwargs is None :
                    return optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer, T_0=self.scheduler_period, T_mult=1, eta_min=0)
                else :
                    return optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer, **kwargs)

            elif scheduler_name == "CyclicLR" : #Seems to maximize at 1/3rd of the steps
                if kwargs is None :
                    return optim.lr_scheduler.CyclicLR(self.optimizer, base_lr=self.lr, max_lr=self.max_lr, step_size_up=5, mode="exp_range", gamma=0.85)
                else :
                    return optim.lr_scheduler.CyclicLR(self.optimizer, **kwargs)

            elif scheduler_name == "OneCycleLR" :
                if kwargs is None :
                    return optim.lr_scheduler.OneCycleLR(self.optimizer, max_lr=self.max_lr, total_steps=num_steps)
                else :
                    return optim.lr_scheduler.OneCycleLR(self.optimizer, **kwargs)

            elif scheduler_name == "ReduceLROnPlateau" :
                if kwargs is None :

                    return custom_reduce_on_plateau(self.optimizer, score_check=self.get_last_score)
                else :
                    return custom_reduce_on_plateau(self.optimizer, **kwargs)

            elif scheduler_name == "SWA" :
                if kwargs is None :
                    return optim.swa_utils.SWALR(self.optimizer, anneal_strategy="cos", anneal_epochs=num_steps, swa_lr=self.max_lr)
                else :
                    return optim.swa_utils.SWALR(self.optimizer, anneal_epochs=self.SWALR_epochs, swa_lr=self.max_lr, **kwargs)

        # create batch/epoch lookup

        # Warmup
        if self.warmup_batches>0 :
            schedulers.append(create_scheduler('warmup', self.warmup_batches))
            # print('warmup', self.warmup_batches, "@", self.warmup_batches)

            # scheduler_steps.insert(0, self.warmup_batches)
            self.scheduler_checkpoints.append(self.warmup_batches)

        for scheduler_name, steps, kwargs in zip(scheduler_names, scheduler_steps, kwarg_list):
            if scheduler_name is "OneCycleLR" :
                steps=steps*self.batches_per_epoch

            schedulers.append(create_scheduler(scheduler_name, steps, kwargs))

            if len(self.scheduler_checkpoints)>=1 :
                self.scheduler_checkpoints.append(steps+self.scheduler_checkpoints[-1])
            else :
                self.scheduler_checkpoints.append(steps)

        if self.SWA :
            if "SWA" not in self.scheduler_list : #If not already added manually
                schedulers.append(create_scheduler('SWA', (self.SWALR_epochs)))
                # print('SWA', self.SWALR_epochs, "@", self.SWALR_epochs+(self.scheduler_checkpoints[-1]))
                self.scheduler_checkpoints.append(self.SWALR_epochs+(self.scheduler_checkpoints[-1]))

        print(schedulers)
        if len(schedulers)==1 :
            return schedulers[0]
        else :
            # del self.scheduler_checkpoints[-1] # Does not want the last checkpoint
            return optim.lr_scheduler.SequentialLR(self.optimizer, schedulers, self.scheduler_checkpoints[:-1])

    # Wraps and handles the sequential scheduler to ensure it is being called at the correct locale
    def scheduler_step(self, step, call_place='epoch', val_loss=None) :

        if self.num_steps>len(self.scheduler_list)-1 :
            curr_scheduler=self.scheduler_list[-1]
        else :
            curr_scheduler=self.scheduler_list[self.num_steps-1]
        # Schedulers that are called on a per batch basis
        if call_place=='batch' :
            if curr_scheduler=="warmup" :
                self.num_steps+=1
                self.scheduler.step()
            elif curr_scheduler=="OneCycleLR" :
                self.num_steps+=1
                self.scheduler.step()

        # Schedulers that are called on a per accumulate basis
        # https://discuss.pytorch.org/t/gradient-accumulation-and-scheduler/69077
        elif call_place=='accumulate' :
            if curr_scheduler=="CosineAnnealingLR" :
                print(curr_scheduler)
                self.num_steps+=1
                self.scheduler.step()

            elif curr_scheduler=="CosineAnnealingWarmRestarts" :
                print(curr_scheduler)
                self.num_steps+=1
                self.scheduler.step()
            elif curr_scheduler=="CyclicLR" :
                print(curr_scheduler)
                self.num_steps+=1
                self.scheduler.step()



            elif curr_scheduler=="batch" : # Catch all for other batch updating schedulers
                print(curr_scheduler)
                self.num_steps+=1
                self.scheduler.step()

        # Schedulers that are called once per epoch
        elif call_place=='epoch' :

            if curr_scheduler=="ReduceLROnPlateau" :
                print(curr_scheduler)
                self.num_steps+=1
                self.scheduler.step()

            elif curr_scheduler=="SWA" :
                print(curr_scheduler)
                self.num_steps+=1
                self.SWA_active=True
                self.scheduler.step()
                self.epochs_since_improvement=0
                self.SWA_model.update_parameters(self.model)

            elif curr_scheduler=="epoch" : # Catch all for other epoch updating schedulers
                print(curr_scheduler)
                self.num_steps+=1
                self.scheduler.step()

            # wm=self.scheduler_list.count("warmup")
            # car=self.scheduler_list.count("CosineAnnealingWarmRestarts")
            # rop=self.scheduler_list.count("ReduceLROnPlateau")
            # print("checkpoints",self.scheduler_checkpoints)
            # print("counts", wm, car, rop)
            # print("added counts",wm, wm+car, wm+car+rop)
            # print("input step",step)
            # print("count step",self.num_steps)

    # This is just a normal forward pass, but it is split out to allow for the generalizability of the code and custom functions to be passed in
    def standard_forward(self, model, data) :
        input, values=data
        input, values=input.to(self.device), values.to(self.device)
        output=self.model(input)

        loss=self.criterion(output, values)
        return loss

    def standard_backward(self, loss) :
        loss.backward()

    def standard_validation(self, model, dataloader, validate_accuracy) :
        running_loss=0.0
        total=0
        score=0.0
        with torch.no_grad():
            for i, data in enumerate(dataloader):
                input, values=data
                input, values=input.to(self.device), values.to(self.device)

                output=self.model(input)

                if validate_accuracy :
                    # Returns the indecies of the max value on the first dimension. e.g. which class has the max value
                    _, predicted = torch.max(output.data, dim=1)

                    # Total number of runs
                    total += values.size(0)

                    # Total number of instances where the max value index matches the correct class
                    score += (predicted == values).sum().item()

                loss=self.criterion(output, values).item()
                running_loss+=loss

        if self.validate_accuracy :
            return running_loss/(i+1), score/total
        else :
            return running_loss/(i+1)

    def load_checkpoint(self, checkpoint_name="best.pt", checkpoint_dict=None) :
        if checkpoint_dict is None :
            checkpoint_dict=torch.load(self.checkpoint_dir+"/"+checkpoint_name)

        self.model.load_state_dict(checkpoint_dict["model_params"])
        self.optimizer.load_state_dict(checkpoint_dict["optimizer"])
        self.SWA_active=checkpoint_dict["SWA_active"]
        if self.SWA_active :
            self.SWA_model.load_state_dict(checkpoint_dict["swa"])

        self.last_epoch=checkpoint_dict["epoch"]
        self.best_score=checkpoint_dict["score"]

    def standard_early_stopping(self, score, epoch, maximize=False) :
        improved=False

        if (maximize and score>self.best_score) :
            improved=True
        elif (not maximize and score<self.best_score) :
            improved=True
        print(score)
        if improved :
            print("Improved by {}".format(abs(self.best_score-score)))
            self.best_score=score
            self.epochs_since_improvement=0
            checkpoint={"model_params":self.model.state_dict(), #Use passed in model in the case of SWA
                        "optimizer": self.optimizer.state_dict(),
                        "epoch":epoch,
                        "swa": self.SWA_model.state_dict(),
                        "score":score,
                        "SWA_active":self.SWA_active
                        }

            torch.save(checkpoint, self.checkpoint_dir+"best.pt")
            # torch.save(self.model.state_dict(), self.checkpoint_dir+"E-{}_S-{}".format(epoch, score))
        else :
            self.epochs_since_improvement+=1
            if self.epochs_since_improvement>=self.early_stopping_patience :
                self.load_checkpoint()
                print("Model not improved in {} epochs".format(self.epochs_since_improvement))
                print("Training complete: Best performance at epoch {} with a score of {}".format(self.last_epoch, self.best_score))

                if self.SWA_active : #Updated to state of load_checkpoint
                    return self.SWA_model
                else :
                    return self.model

    def standard_function_end(self, epoch, train_loss, val_loss=None, accuracy=None) :
        if accuracy is not None :
            print("Epoch {}:".format(epoch))

            print("\tTrain loss\t{:4f}".format(train_loss))
            self.writer.add_scalar("Train loss", train_loss, epoch)

            if val_loss is not None :
                print("\tVal loss\t{}".format(val_loss))
                self.writer.add_scalar("Val loss", val_loss, epoch)

            if accuracy is not None :
                print("\tAccuracy\t{:2f}".format(accuracy*100))
                self.writer.add_scalar("Accuracy", accuracy*100, epoch)

            self.writer.add_scalar("LR",self.optimizer.param_groups[0]["lr"], epoch)

        print()



    def train(self, epochs=10) :
            self.ASAM_flip_flop=1 #Set for ascent step
            val_loss=None
            last_opt_step=0

            for epoch in range(self.last_epoch, epochs) :
                epoch_loss=0.0

                #init dataloaders
                if self.loader_kwargs is not None :
                    self.train_loader=torch.utils.data.DataLoader(self.train_dataset, **self.loader_kwargs)
                    self.val_loader=torch.utils.data.DataLoader(self.val_dataset, **self.loader_kwargs)
                else :
                    self.train_loader=torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.loader_workers, shuffle=self.shuffle_loader)
                    self.val_loader=torch.utils.data.DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.loader_workers, shuffle=self.shuffle_loader)


                self.last_epoch=epoch #for resuming
                i=0
                for i, data in enumerate(self.train_loader, 0):
                # for data in self.train_loader:
                    # i+=1

                    total_batches= i + self.batches_per_epoch*epoch

                    # Train
                    train_loss=self.forward_model(self.model, data)

                    self.backward(train_loss)

                    if self.ASAM_flip_flop : #If ASAM, only counts loss from before ascent step. else, always counts loss
                        train_loss+=train_loss.item()

                    # Warmup and per batch schedulers
                    self.scheduler_step(total_batches, call_place='batch')

                    #Step optimizer
                    if total_batches - last_opt_step >= self.accumulate_batches:

                        if self.clip_grad_norm :
                            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)

                        # # Optimizer step
                        # if self.SAM :
                        #     # self.optimizer.step(closure) #possibly depricated
                        #     self.minimizer.ascent_step() # contains zero_grad

                        if self.ASAM or self.SAM:
                            if self.ASAM_flip_flop : # Case one : ascent step
                                print("ascent")
                                self.minimizer.ascent_step() # contains zero_grad
                            else :
                                print("descent")

                                self.minimizer.descent_step() # contains zero_grad

                            self.ASAM_flip_flop=1-self.ASAM_flip_flop #switch between 1 and 0

                        else :
                            self.optimizer.step()
                            self.optimizer.zero_grad()


                        self.scheduler_step(total_batches, call_place='accumulate')
                        last_opt_step = total_batches

                        #Do something with training loss
                        epoch_loss+=train_loss
                        train_loss=0

                ## End of Epoch ##
                # Validate
                accuracy=None
                if self.validate :
                    score=None
                #     val_loss, acc= self.val_function(self.model, self.val_loader, self.accuracy)

                    val_model=self.model

                    #Validate using SWA model if it has been updated
                    if self.SWA_active:
                        val_model=self.SWA_model

                    if self.validate_accuracy :
                        val_loss, acc= self.val_function(val_model, self.val_loader, self.validate_accuracy)
                        accuracy=acc
                        score=acc
                    else :
                        val_loss= self.val_function(val_model, self.val_loader, self.validate_accuracy)
                        score=val_loss
                    self.last_score=score
                    best_model=None
                    if epoch>self.warmup_epochs :
                        best_model=self.early_stopping_function(score, epoch, maximize=self.validate_accuracy)
                    else :
                        self.epochs_since_improvement=0



                    if best_model is not None:
                        # print("Training complete after {} Epochs")
                        if self.SWA_active :
                            torch.optim.swa_utils.update_bn(self.train_loader, self.SWA_model)
                        return best_model #Is SWA model if SWA was active at the time
                self.end_of_epoch_function(epoch, epoch_loss/(self.batches_per_epoch*self.batch_size), val_loss, accuracy)
                # Update scheduler
                self.scheduler_step(total_batches, call_place='epoch', val_loss=val_loss)
                #
                # with tune.checkpoint_dir(epoch) as checkpoint_dir:
                #     path = os.path.join(checkpoint_dir, "checkpoint")
                #     torch.save((model.state_dict(), optimizer.state_dict()), path)
            return self.model
