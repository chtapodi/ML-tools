import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms
from torch import nn
from torch import optim
import torchvision.utils
import matplotlib.pyplot as plts
from ASAM.asam import ASAM, SAM

class custom_warmup :
    #Adapted from yolov5 code
    def __init__(self, optimizer, num_warmup, lr=.01, warmup_bias_lr=.1, momentum=0.937, warmup_momentum=0.8) :
        self.optimizer=optimizer
        self.num_warmup=num_warmup
        self.warmup_bias_lr=warmup_bias_lr
        self.momentum=momentum
        self.warmup_momentum=warmup_momentum

        self.batches=0
        self.lf=  lambda x: (1 - x / epochs) * (1.0 - hyp['lrf']) + hyp['lrf']  # linear

    def step(self, epoch) :
        self.batches+=1
        xi = [0, nw]  # x interp
        # accumulate = max(1, np.interp(ni, xi, [1, nbs / batch_size]).round())
        for j, x in enumerate(self.optimizer.param_groups):
            # bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
            x['lr'] = np.interp(self.batches, xi, [self.warmup_bias_lr if j == 0 else 0.0, x['initial_lr'] * self.lf(epoch)])
            if 'momentum' in x:
                x['momentum'] = np.interp(self.batches, xi, [self.warmup_momentum, self.momentum])



class trainer :
    def __init(self, model, train_dataset, val_dataset, batch_size=16, val_dataset=None,
    forward_function=None, val_function=None, report_function=None,
    optimizer='SGD', warmup_scheduler=None, warmup_epochs=5, accumulate_batches=1,
    scheduler_names=[], scheduler_checkpoints=[]
    SAM=False, ASAM=False, SWA=False, SWALR_epochs=5, SWA_lr=.05
    lr=.01, momentum=0.937, scheduler_period=10,
    checkpoint_dir="training_checkpoints/") :

        # Scalers
        self.lr=lr
        self.momentum=momentum



        self.model=model

        #init dataloaders
        self.train_loader=torch.utils.data.DataLoader(train_dataset)
        self.val_loader=torch.utils.data.DataLoader(val_dataset)

        self.batches_per_epoch=len(self.train_loader)
        self.accumulate_batches=accumulate_batches

        self.optimizer=optimizer
        if isinstance(self.optimizer, str): #If optimizer name passed in, generate optimizer
            self.optimizer=self.generate_optimizer(self.optimizer, momentum)


        self.warmup_scheduler=warmup_scheduler
        self.warmup_batches=warmup_epochs*self.batches_per_epoch

        if self.warmup_scheduler is None :
            self.warmup_scheduler=custom_warmup



        self.SWA=SWA
        self.SWALR_epochs=SWALR_epochs
        self.SWA_lr=SWA_lr
        if self.SWA :
            self.swa_model = torch.optim.swa_utils.AveragedModel(self.model)



        self.scheduler_period=scheduler_period #for cosine schedulers
        self.scheduler=self.generate_scheduler(scheduler_names, scheduler_checkpoints)


        self.forward_function=forward_function

        self.val_function=val_function


    def generate_optimizer(self, optimizer_name, momentum) :

        params=self.model.parameters()
        lr=self.lr
        optimizer=None
        optimizer_constructor=None

        if "Adam" in optimizer_name :
            if optimizer_name is "Adam" :
                optimizer_constructor=torch.optim.Adam

            elif optimizer_name is "AdamW" :
                optimizer_constructor=torch.optim.AdamW

            if self.SAM :
                optimizer=SAM(params, optimizer_constructor, lr=lr, betas=(momentum, 0.999), weight_decay=self.decay)
            elif self.ASAM :
                optimizer=ASAM(params, optimizer_constructor, lr=lr, betas=(momentum, 0.999), weight_decay=self.decay)

            else :
                optimizer=optimizer_constructor(params, lr=lr, betas=(momentum, 0.999), weight_decay=self.decay)
        else :

            if optimizer_name is "RMSProp" :
                optimizer_constructor=torch.optim.RMSProp
                if self.SAM :
                    optimizer=SAM(params, optimizer_constructor, lr=lr, momentum=momentum, weight_decay=self.decay)
                elif self.ASAM :
                    optimizer=ASAM(params, optimizer_constructor, lr=lr, momentum=momentum, weight_decay=self.decay)
                else :
                    optimizer=optimizer_constructor(params, lr=lr, momentum=momentum, weight_decay=self.decay)

            else :
                optimizer_constructor=torch.optim.SGD #Default option
                if self.SAM :
                    optimizer=SAM(params, optimizer_constructor, lr=lr, momentum=momentum, nesterov=True, weight_decay=self.decay)
                elif self.ASAM :
                    optimizer=ASAM(params, optimizer_constructor, lr=lr, momentum=momentum, nesterov=True, weight_decay=self.decay)
                else :
                    optimizer=optimizer_constructor(params, lr=lr, momentum=momentum, nesterov=True, weight_decay=self.decay)

        return optimizer

    def generate_scheduler(self, scheduler_names, scheduler_checkpoints) :
        # https://www.kaggle.com/code/isbhargav/guide-to-pytorch-learning-rate-scheduling/notebook
        # Scheduler checkpoints is the last step the associated scheduler should run at
        def create_scheduler(scheduler_name, num_steps, strategy="") :
            # TODO: update specific parameters, make them configurable
                for i in num_steps :
                    self.scheduler_list.append(scheduler_name)

            if scheduler_name is "warmup" :
                return self.warmup_scheduler(optimizer, num_steps)

            elif scheduler_name is "CosineAnnealingLR" :
                return torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.scheduler_period)

            elif scheduler_name is "CosineAnnealingWarmRestarts" :
                return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer, T_0=self.scheduler_period, T_mult=1, eta_min=0)

            elif scheduler_name is "CyclicLR" :
                if strategy is "" :
                    strategy="triangular"
                return torch.optim.lr_scheduler.CyclicLR(self.optimizer, base_lr=self.lr, max_lr=0.1, step_size_up=5, mode=strategy, gamma=0.85)

            elif scheduler_name is "OneCycleLR" :
                return torch.optim.lr_scheduler.OneCycleLR(self.optimizer, max_lr=0.01, steps_per_epoch=len(data_loader), epochs=10)

            elif scheduler_name is "ReduceLROnPlateau" :
                return torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer)

            elif scheduler_name is "SWA" :

                return swa_scheduler=torch.optim.swa_utils.SWALR(optimizer, anneal_strategy=strategy, anneal_epochs=SWALR_epochs, swa_lr=self.SWA_lr)

        # create batch/epoch lookup
        schedulers=[]

        # Warmup
        if self.warmup_batches>0 :
            schedulers.append(create_scheduler('warmup', self.warmup_batches))
            scheduler_checkpoints.insert(0, self.warmup_batches)

        for scheduler_name, checkpoint in zip(scheduler_names, scheduler_checkpoints):
            schedulers.append(create_scheduler(scheduler_name, checkpoint-scheduler_checkpoints[-1]))

        if self.SWA :
            schedulers.append(create_scheduler('SWA', checkpoint+scheduler_checkpoints[-1]))
            scheduler_checkpoints.append(self.SWALR_epochs)

        self.scheduler=torch.optim.lr_scheduler.SequentialLR(self.optimizer, schedulers, scheduler_checkpoints)

        # Wraps and handles the sequential scheduler to ensure it is being called at the correct locale
    def scheduler_step(self, epoch, call_place='epoch', val_loss=None) :
            curr_scheduler=self.scheduler_list[epoch]

            # Schedulers that are called on a per batch basis
            if call_place=='batch' :
                if curr_scheduler=="warmup" :
                    self.scheduler.step(epoch)
                else :
                    print("Something is not right with scheduler setup")
                    print("{} stepped in {}".format(curr_scheduler, call_place))

            # Schedulers that are called on a per accumulate basis
            # https://discuss.pytorch.org/t/gradient-accumulation-and-scheduler/69077
            elif call_place=='accumulate' :
                if curr_scheduler=="CosineAnnealingWarmRestarts" :
                    self.scheduler.step()

                elif curr_scheduler=="CyclicLR" :
                    self.scheduler.step()

                elif curr_scheduler=="OneCycleLR" :
                    self.scheduler.step()

                elif curr_scheduler=="batch" : # Catch all for other batch updating schedulers
                    self.scheduler.step()
                else :
                    print("Something is not right with scheduler setup")
                    print("{} stepped in {}".format(curr_scheduler, call_place))

            # Schedulers that are called once per epoch
            elif call_place=='epoch' :
                if curr_scheduler=="ReduceLROnPlateau" :
                    self.scheduler.step(val_loss)

                elif curr_scheduler=="SWA" :
                    self.scheduler.step()
                    self.swa_model.update_parameters(self.model)

                elif curr_scheduler=="epoch" : # Catch all for other epoch updating schedulers
                    self.scheduler.step()
                else :
                    print("Something is not right with scheduler setup")
                    print("{} stepped in {}".format(curr_scheduler, call_place))

    # This is just a normal forward pass, but it is split out to allow for the generalizability of the code and custom functions to be passed in
    def standard_forward(self, model, data) :
        input, values=data
        input, values=input.to(self.device), values.to(self.device)

        output=self.model(input)

        loss=self.criterion(output, values)
        return loss


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

            for epoch in epochs(self.last_epoch, epochs) :
                self.last_epoch=epoch #for resuming

                for i, data in enumerate(self.train_loader, 0):

                    total_batches= i + self.batches_per_epoch*epoch

                    # Train
                    train_loss=self.forward_model(self.model, data)

                    train_loss.backwards()

                    if self.ASAM_flip_flop : #If ASAM, only counts loss from before ascent step. else, always counts loss
                        train_loss+=train_loss.item()

                    # Warmup and per batch schedulers
                    self.scheduler_step(epoch, call_pos='batch')


                    #Step optimizer
                    if total_batches - last_opt_step >= self.accumulate_batches:

                        if self.clip_grad_norm :
                            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)

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
                            optimizer.zero_grad()


                        self.scheduler_step(epoch, call_pos='accumulate')

                        last_opt_step = total_batches

                        #Do something with training loss

                        train_loss=0


                ## End of Epoch ##

                # Validate
                if self.validate :
                    val_loss, acc= self.val_function(self.model, self.val_loader, self.accuracy)


                # Update scheduler
                self.scheduler_step(epoch, call_pos='epoch', val_loss)


                with tune.checkpoint_dir(epoch) as checkpoint_dir:
                    path = os.path.join(checkpoint_dir, "checkpoint")
                    torch.save((model.state_dict(), optimizer.state_dict()), path)
