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

    def step(self) :
        self.batches+=1
        xi=[0, self.num_warmup]
        # accumulate = max(1, np.interp(ni, xi, [1, nbs / batch_size]).round())
        bias_lr=np.interp(self.batches, xi, [self.warmup_bias_lr, self.lr])
        lr=np.interp(self.batches, xi, [0.0, self.lr])
        self.last_lr=lr
        momentum=np.interp(self.batches, xi, [self.warmup_momentum, self.momentum])

        for j, x in enumerate(self.optimizer.param_groups):
            # bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
            if j==0 :
                x['lr']=bias_lr
            else :
                x['lr'] = lr

            if 'momentum' in x:
                x['momentum'] = momentum

    def get_last_lr(self) :
        return self.last_lr


class trainer :
    def __init__(self, model, train_dataset, val_dataset=None,
        loader_kwargs=None, loader_workers=4, batch_size=16, shuffle_loader=True,
        forward_function=None, backward_function=None, clip_grad_norm=True,
        val_function=None, report_function=None, criterion=None,
        optimizer='SGD', warmup_scheduler=None, warmup_epochs=5, accumulate_batches=1,
        use_discriminativeLR=False, discriminativeLR=.0001, forward_model=None,
        scheduler_names=[], scheduler_steps=[], scheduler_kwargs=None,
        SAM=False, ASAM=False, SWA=False, SWALR_epochs=5, SWA_lr=.05,
        lr=.01, decay=0.0005, momentum=0.937, scheduler_period=10,
        device=None, checkpoint_dir="training_checkpoints/") :
        print("initch",scheduler_steps)

        self.device=device
        if self.device is None :
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Scalers
        self.lr=lr
        self.momentum=momentum
        self.decay=decay

        self.SAM=SAM

        self.ASAM=ASAM
        if self.ASAM and self.SAM :
            printf("SAM and ASAM cannot be enabled at the same time")
            exit()


        self.model=model
        self.model.to(self.device)

        self.criterion=criterion
        if self.criterion is None :
            self.criterion=torch.nn.MSELoss()

        # if discriminativeLR :
        #     params, lr_arr, _ = dlr.discriminative_lr_params(self.model, slice(min_lr, self.lr)) #slice(min_lr,max_lr)
        #     for p in params :
        #         p['lr']=(float)(p['lr'])

        self.loader_kwargs=loader_kwargs
        self.loader_workers=loader_workers
        self.train_dataset=train_dataset
        self.val_dataset=val_dataset
        self.batch_size=batch_size
        self.shuffle_loader=shuffle_loader
        self.clip_grad_norm=clip_grad_norm

        self.batches_per_epoch=len(self.train_dataset)
        self.accumulate_batches=accumulate_batches

        self.SWA=SWA
        self.SWALR_epochs=SWALR_epochs
        self.SWA_lr=SWA_lr
        if self.SWA :
            self.swa_model = optim.swa_utils.AveragedModel(self.model)

        self.optimizer=optimizer
        if isinstance(self.optimizer, str): #If optimizer name passed in, generate optimizer
            self.optimizer=self.generate_optimizer(self.optimizer, momentum)


        self.warmup_scheduler=warmup_scheduler
        self.warmup_batches=warmup_epochs*self.batches_per_epoch

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

        self.val_function=val_function


        self.last_epoch=0


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

            if self.SAM :
                optimizer=SAM(params, optimizer_constructor, lr=lr, betas=(momentum, 0.999), weight_decay=self.decay)
            elif self.ASAM :
                optimizer=ASAM(params, optimizer_constructor, lr=lr, betas=(momentum, 0.999), weight_decay=self.decay)

            else :
                optimizer=optimizer_constructor(params, lr=lr, betas=(momentum, 0.999), weight_decay=self.decay)
        else :

            if optimizer_name == "RMSProp" :
                optimizer_constructor=optim.RMSProp
                if self.SAM :
                    optimizer=SAM(params, optimizer_constructor, lr=lr, momentum=momentum, weight_decay=self.decay)
                elif self.ASAM :
                    optimizer=ASAM(params, optimizer_constructor, lr=lr, momentum=momentum, weight_decay=self.decay)
                else :
                    optimizer=optimizer_constructor(params, lr=lr, momentum=momentum, weight_decay=self.decay)

            else :
                optimizer_constructor=optim.SGD #Default option
                if self.SAM :
                    optimizer=SAM(params, optimizer_constructor, lr=lr, momentum=momentum, nesterov=True, weight_decay=self.decay)
                elif self.ASAM :
                    optimizer=ASAM(params, optimizer_constructor, lr=lr, momentum=momentum, nesterov=True, weight_decay=self.decay)
                else :
                    optimizer=optimizer_constructor(params, lr=lr, momentum=momentum, nesterov=True, weight_decay=self.decay)

        return optimizer

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

            elif scheduler_name == "CyclicLR" :
                if kwargs is None :
                    return optim.lr_scheduler.CyclicLR(self.optimizer, base_lr=self.lr, max_lr=0.1, step_size_up=5, mode="triangular2", gamma=0.85)
                else :
                    return optim.lr_scheduler.CyclicLR(self.optimizer, **kwargs)

            elif scheduler_name == "OneCycleLR" :
                if kwargs is None :
                    return optim.lr_scheduler.OneCycleLR(self.optimizer, max_lr=.1, total_steps=num_steps)
                else :
                    return optim.lr_scheduler.OneCycleLR(self.optimizer, **kwargs)

            elif scheduler_name == "ReduceLROnPlateau" :
                if kwargs is None :
                    return optim.lr_scheduler.ReduceLROnPlateau(self.optimizer)
                else :
                    return optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, **kwargs)

            elif scheduler_name == "SWA" :
                if kwargs is None :
                    return optim.swa_utils.SWALR(self.optimizer, anneal_strategy="cos", anneal_epochs=self.SWALR_epochs, swa_lr=self.SWA_lr)
                else :
                    return optim.swa_utils.SWALR(self.optimizer, anneal_epochs=self.SWALR_epochs, swa_lr=self.SWA_lr, **kwargs)

        # create batch/epoch lookup

        # Warmup
        if self.warmup_batches>0 :
            schedulers.append(create_scheduler('warmup', self.warmup_batches))
            # scheduler_steps.insert(0, self.warmup_batches)
            self.scheduler_checkpoints.append(self.warmup_batches)

        for scheduler_name, steps, kwargs in zip(scheduler_names, scheduler_steps, kwarg_list):
            schedulers.append(create_scheduler(scheduler_name, steps, kwargs))
            if len(self.scheduler_checkpoints)>=1 :
                self.scheduler_checkpoints.append(self.scheduler_checkpoints[-1]+steps)
            else :
                self.scheduler_checkpoints.append(steps)


        if self.SWA :
            schedulers.append(create_scheduler('SWA', self.SWALR_epochs))
            self.scheduler_checkpoints.append(self.SWALR_epochs+self.scheduler_checkpoints[-1])



        if len(schedulers)==1 :
            return schedulers[0]
        else :
            # del self.scheduler_checkpoints[-1] # Does not want the last checkpoint
            return optim.lr_scheduler.SequentialLR(self.optimizer, schedulers, self.scheduler_checkpoints[:-1])

    # Wraps and handles the sequential scheduler to ensure it is being called at the correct locale
    def scheduler_step(self, epoch, call_place='epoch', val_loss=None) :
            curr_scheduler=self.scheduler_list[epoch]

            # Schedulers that are called on a per batch basis
            if call_place=='batch' :
                if curr_scheduler=="warmup" :
                    self.scheduler.step()
                # else :
                #     print("Something is not right with scheduler setup")
                #     print("{} stepped in {}".format(curr_scheduler, call_place))

            # Schedulers that are called on a per accumulate basis
            # https://discuss.pytorch.org/t/gradient-accumulation-and-scheduler/69077
            elif call_place=='accumulate' :
                if curr_scheduler=="CosineAnnealingLR" :
                    self.scheduler.step()

                elif curr_scheduler=="CosineAnnealingWarmRestarts" :
                    self.scheduler.step()
                elif curr_scheduler=="CyclicLR" :
                    self.scheduler.step()

                elif curr_scheduler=="OneCycleLR" :
                    self.scheduler.step()

                elif curr_scheduler=="batch" : # Catch all for other batch updating schedulers
                    self.scheduler.step()
                # else :
                #     print("Something is not right with scheduler setup")
                #     print("{} stepped in {}".format(curr_scheduler, call_place))

            # Schedulers that are called once per epoch
            elif call_place=='epoch' :
                if curr_scheduler=="ReduceLROnPlateau" :
                    self.scheduler.step(val_loss)

                elif curr_scheduler=="SWA" :
                    self.scheduler.step()
                    self.swa_model.update_parameters(self.model)

                elif curr_scheduler=="epoch" : # Catch all for other epoch updating schedulers
                    self.scheduler.step()
                # else :
                #     print("Something is not right with scheduler setup")
                #     print("{} stepped in {}".format(curr_scheduler, call_place))

    # This is just a normal forward pass, but it is split out to allow for the generalizability of the code and custom functions to be passed in
    def standard_forward(self, model, data) :
        input, values=data
        input, values=input.to(self.device), values.to(self.device)
        output=self.model(input)

        loss=self.criterion(output, values)
        return loss

    def standard_backward(self, loss) :
        loss.backward()


    def validate(self, model, dataloader, get_accuracy=False) :
        running_loss=0.0
        total=0
        with torch.no_grad():
            for i, data in enumerate(dataloader):
                input, values=data
                input, values=input.to(self.device), values.to(self.device)

                output=self.model(input)

                if get_accuracy :
                    # Returns the indecies of the max value on the first dimension. e.g. which class has the max value
                    _, predicted = torch.max(outputs.data, dim=1)

                    # Total number of runs
                    total += labels.size(0)

                    # Total number of instances where the max value index matches the correct class
                    score += (predicted == labels).sum().item()

                loss=self.criterion(output, values).item()

        if get_accuracy :
            return running_loss/(i+1), score/total
        else :
            return running_loss/(i+1), 0


    #TODO make this better
    def epoch_end(self, epoch, train_loss, val_loss, acc=0) :
        if acc==0 :
            print("Epoch {}:\t train loss: {:.3f}\t val_loss:{:.3f}".format(epoch, train_loss, val_loss))
        else :
            print("Epoch {}:\t train loss: {:.3f}\t val_loss:{:.3f} \t acc:{:.2f}".format(epoch, train_loss, val_loss, acc*100))

        if self.custom_report is not None :
            self.custom_report(epoch, train_loss, val_loss, acc)

    def train(self, epochs=10) :
            self.ASAM_flip_flop=1 #Set for ascent step
            val_loss=0
            last_opt_step=0

            for epoch in range(self.last_epoch, epochs) :
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
                    i+=1


                    total_batches= i + self.batches_per_epoch*epoch

                    # Train
                    train_loss=self.forward_model(self.model, data)

                    self.backward(train_loss)

                    if self.ASAM_flip_flop : #If ASAM, only counts loss from before ascent step. else, always counts loss
                        train_loss+=train_loss.item()

                    # Warmup and per batch schedulers
                    self.scheduler_step(epoch, call_place='batch')


                    #Step optimizer
                    if total_batches - last_opt_step >= self.accumulate_batches:

                        if self.clip_grad_norm :
                            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)

                        # Optimizer step
                        if self.SAM :
                            # self.optimizer.step(closure) #possibly depricated
                            optimizer.ascent_step() # contains zero_grad

                        elif self.ASAM :
                            if self.ASAM_flip_flop : # Case one : ascent step
                                optimizer.ascent_step() # contains zero_grad
                            else :
                                optimizer.descent_step() # contains zero_grad

                            self.ASAM_flip_flop=1-self.ASAM_flip_flop #switch between 1 and 0

                        else :
                            self.optimizer.step()
                            self.optimizer.zero_grad()


                        self.scheduler_step(epoch, call_place='accumulate')
                        last_opt_step = total_batches

                        #Do something with training loss
                        train_loss=0

                ## End of Epoch ##

                # Validate
                # if self.validate :
                #     val_loss, acc= self.val_function(self.model, self.val_loader, self.accuracy)

                # Update scheduler
                self.scheduler_step(epoch, call_place='epoch', val_loss=val_loss)
                # 
                # with tune.checkpoint_dir(epoch) as checkpoint_dir:
                #     path = os.path.join(checkpoint_dir, "checkpoint")
                #     torch.save((model.state_dict(), optimizer.state_dict()), path)
