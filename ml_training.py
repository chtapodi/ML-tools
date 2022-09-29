class custom_warmup :
    def __init(self, params, num_warmup, )




class trainer :
    def __init(self, optimizer, SWA, optimizer_name, momentum, checkpoint_dir="training_checkpoints/")

        if optimizer is None :
            self.optimizer=self.generate_optimizer(optimizer_name, momentum)




        self.batches_per_epoch=len(train_loader)


        if SWA :
            self.swa_model = torch.optim.swa_utils.AveragedModel(model)

    def generate_optimizer(self, optimizer_name, momentum) :

        lr=self.lr
        optimizer=None
        optimizer_constructor=None

        if "Adam" in config["optimizer_name"] :
            if optimizer_name is "Adam" :
                optimizer_constructor=torch.optim.Adam

            elif optimizer_name is "AdamW" :
                optimizer_constructor=torch.optim.AdamW

            if self.SAM :
                optimizer=SAM(params, optimizer_constructor, lr=lr, betas=(momentum, 0.999), weight_decay=self.decay)
            else :
                optimizer=optimizer_constructor(params, lr=lr, betas=(momentum, 0.999), weight_decay=self.decay)
        else :

            if optimizer_name is "RMSProp" :
                optimizer_constructor=torch.optim.RMSProp
                if self.SAM :
                    optimizer=SAM(params, optimizer_constructor, lr=lr, momentum=momentum, weight_decay=self.decay)
                else :
                    optimizer=optimizer_constructor(params, lr=lr, momentum=momentum, weight_decay=self.decay)

            else :
                optimizer_constructor=torch.optim.SGD #Default option
                if self.SAM :
                    optimizer=SAM(params, optimizer_constructor, lr=lr, momentum=momentum, nesterov=True, weight_decay=self.decay)
                else :
                    optimizer=optimizer_constructor(params, lr=lr, momentum=momentum, nesterov=True, weight_decay=self.decay)

        return optimizer

    def generate_scheduler(self, scheduler_names, warmup_batches, warmup_strategy, scheduler_checkpoints) :
        # TODO: yolov5 has bias fall from .1 to lr and everything else raise from 0 to lr, look into if this is worth implementing
        # https://www.kaggle.com/code/isbhargav/guide-to-pytorch-learning-rate-scheduling/notebook
        def create_scheduler(scheduler_name, num_epochs, strategy="") :
            # TODO: update specific parameters, make them configurable
                for i in num_epochs :
                    self.scheduler_list.append(scheduler_name)

            if scheduler_name is "warmup" :
                return torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.scheduler_period)

            elif scheduler_name is "CosineAnnealingLR" :
                return torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.scheduler_period)

            elif scheduler_name is "CosineAnnealingWarmRestarts" :
                return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer, T_0=self.scheduler_period, T_mult=1, eta_min=0)

            elif scheduler_name is "CyclicLR" :
                return torch.optim.lr_scheduler.CyclicLR(self.optimize, base_lr=self.lr, max_lr=0.1, step_size_up=5, mode=strategy, gamma=0.85)

            elif scheduler_name is "OneCycleLR" :
                return torch.optim.lr_scheduler.OneCycleLR(self.optimizer, max_lr=0.01, steps_per_epoch=len(data_loader), epochs=10)

            elif scheduler_name is "ReduceLROnPlateau" :
                return torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, T_max=self.scheduler_period)

            elif scheduler_name is "SWA" :
                return swa_scheduler=torch.optim.swa_utils.SWALR(optimizer, anneal_strategy=SWALR_strategy, anneal_epochs=SWALR_epochs, swa_lr=self.SWA_lr)


        # create batch/epoch lookup


        # Warmup
        if self.warmup_batches>0 :


        # list of schedulers
        # SWA if enabled


        # LR scheduler
        # https://pytorch.org/docs/stable/optim.html#taking-care-of-batch-normalization
        #CosineAnnealingWarmRestarts
        #CyclicLR
            #triangular
            #triangular2
            #exp_range

        if self.SWA :


        # Wraps and handles the sequential scheduler to ensure it is being called at the correct locale
    def scheduler_step(self, epoch, call_place='epoch', val_loss=None) :
            curr_scheduler=self.scheduler_list[epoch]

            # Schedulers that are called on a per batch basis
            if call_place=='batch' :
                if curr_scheduler=="warmup" :
                    self.scheduler.step()
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

    def closure(self) :
        outputs=self.model(inputs)
        loss = self.criterion(outputs, values)
        loss.backward()
        return loss


    def train(self, epochs=10) :


            for epoch in epochs(self.last_epoch, epochs) :
                self.last_epoch=epoch #for resuming

                for i, data in enumerate(, 0):

                    total_batches= i + self.batches_per_epoch*epoch

                    # Train

                    # Warmup and per batch schedulers
                    self.scheduler_step(epoch, call_pos='batch')


                    #Step optimizer
                    if total_batches - last_opt_step >= self.accumulate_batches:

                        if self.clip_grad_norm :
                            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)

                        # Optimizer step

                        if config["SAM"] :
                            self.optimizer.step(closure)
                        else :
                            self.optimizer.step()


                        self.optimizer.zero_grad()

                        self.scheduler_step(epoch, call_pos='accumulate')

                        last_opt_step = total_batches




                ## End of Epoch ##

                # Validate

                # Update scheduler
                self.scheduler_step(epoch, call_pos='epoch', val_loss)


                with tune.checkpoint_dir(epoch) as checkpoint_dir:
                    path = os.path.join(checkpoint_dir, "checkpoint")
                    torch.save((model.state_dict(), optimizer.state_dict()), path)
