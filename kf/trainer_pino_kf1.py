import torch
from torch.cuda import amp
from timeit import default_timer
import pathlib
import sys
import neuralop
import my_tools as myt
import neuralop.mpu.comm as comm

from losses import LpLoss
from neuralop.training.callbacks import PipelineCallback


class Trainer:
    def __init__(self, *, 
                 model, 
                 n_epochs, 
                 wandb_log=True, 
                 device=None, 
                 amp_autocast=False, 
                 callbacks = None,
                 log_test_interval=1, 
                 log_output=False, 
                 use_distributed=False, 
                 verbose=False):
        """
        A general Trainer class to train neural-operators on given datasets

        Parameters
        ----------
        model : nn.Module
        n_epochs : int
        wandb_log : bool, default is True
        device : torch.device
        amp_autocast : bool, default is False
        log_test_interval : int, default is 1
            how frequently to print updates
        log_output : bool, default is False
            if True, and if wandb_log is also True, log output images to wandb
        use_distributed : bool, default is False
            whether to use DDP
        verbose : bool, default is False
        """

        if callbacks:
            assert type(callbacks) == list, "Callbacks must be a list of Callback objects"
            self.callbacks = PipelineCallback(callbacks=callbacks)
            self.override_load_to_device = (self.callbacks.device_load_callback_idx is not None)
            self.overrides_loss = self.callbacks.overrides_loss
        else:
            self.callbacks = []
            self.override_load_to_device = False
            self.overrides_loss = False
        
        if verbose:
            print(f"{self.override_load_to_device=}")
            print(f"{self.overrides_loss=}")

        if self.callbacks:
            self.callbacks.on_init_start(model=model,
                 n_epochs=n_epochs,
                 wandb_log=wandb_log,
                 device=device,
                 amp_autocast=amp_autocast,
                 log_test_interval=log_test_interval,
                 log_output=log_output,
                 use_distributed=use_distributed,
                 verbose=verbose)

        self.model = model
        self.n_epochs = n_epochs

        self.wandb_log = wandb_log
        self.log_test_interval = log_test_interval
        self.log_output = log_output
        self.verbose = verbose
        self.use_distributed = use_distributed
        self.device = device
        self.amp_autocast = amp_autocast

        if self.callbacks:
            self.callbacks.on_init_end(model=model, 
                 n_epochs=n_epochs, 
                 wandb_log=wandb_log, 
                 device=device, 
                 amp_autocast=amp_autocast, 
                 log_test_interval=log_test_interval, 
                 log_output=log_output, 
                 use_distributed=use_distributed, 
                 verbose=verbose)
        
    def train(self, train_loaders, test_loaders,
            optimizer, scheduler, regularizer,
              training_loss=None, eval_losses=None,losstype='sum',check_mem=0):
        
        """Trains the given model on the given datasets.
        params:
        train_loader: torch.utils.data.DataLoader
            training dataloader
        test_loaders: dict[torch.utils.data.DataLoader]
            testing dataloaders
        optimizer: torch.optim.Optimizer
            optimizer to use during training
        optimizer: torch.optim.lr_scheduler
            learning rate scheduler to use during training
        training_loss: training.losses function
            cost function to minimize
        eval_losses: dict[Loss]
            dict of losses to use in self.eval()
        """
        #add here
        use_loader=0
        train_loader=train_loaders[use_loader][0]
        print_normalize=0
        if losstype=='sum':
            print_normalize=sum([i['y'].size(0) for i in train_loader])
        elif losstype=='mean':
            print_normalize=len(train_loader)
        else:
            print('trainer.py,L116: exception loss type')
            exit()
        ##
        test_loaderss={k:v[0] for k,v in test_loaders.items()}
        #  loader has changed inner structure compared with KS experiemtn.
        if self.callbacks:
            self.callbacks.on_train_start(train_loader=train_loader, test_loaders=test_loaders,
                                    optimizer=optimizer, scheduler=scheduler,
                                    regularizer=regularizer, training_loss=training_loss,
                                    eval_losses=eval_losses)
            
        if training_loss is None:
            training_loss = LpLoss(d=2)

        if eval_losses is None: # By default only evaluate on the training_loss
            eval_losses = dict(l2=training_loss)

        if self.use_distributed:
            is_logger = (comm.get_world_rank() == 0)
        else:
            is_logger = True 
        
        for epoch in range(self.n_epochs):

            if self.callbacks:
                self.callbacks.on_epoch_start(epoch=epoch)

            avg_loss = 0
            avg_lasso_loss = 0
            self.model.train()###
            t1 = default_timer()
            train_err = 0.0

            for idx, sample in enumerate(train_loader):###

                if self.callbacks:
                    self.callbacks.on_batch_start(idx=idx, sample=sample)

                # load everything from the batch onto self.device if 
                # no callback overrides default load to device
                
                if self.override_load_to_device:
                    self.callbacks.on_load_to_device(sample=sample)
                else:
                    for k,v in sample.items():
                        if hasattr(v, 'to'):
                            sample[k] = v.to(self.device)

                optimizer.zero_grad(set_to_none=True)
                if regularizer:####
                    regularizer.reset()

                if self.amp_autocast:
                    with amp.autocast(enabled=True):
                        out = self.model(**sample)
                else:
                    out = self.model(**sample)###

                if self.callbacks:
                    self.callbacks.on_before_loss(out=out)

                loss = 0.

                if self.overrides_loss:
                    if isinstance(out, torch.Tensor):
                        loss += self.callbacks.compute_training_loss(out=out.float(), **sample, amp_autocast=self.amp_autocast)
                    elif isinstance(out, dict):
                        loss += self.callbacks.compute_training_loss(**out, **sample, amp_autocast=self.amp_autocast)
                else:

                    if self.amp_autocast:
                        with amp.autocast(enabled=True):
                            if isinstance(out, torch.Tensor):
                                loss = training_loss(out.float(), **sample,t_val=train_loaders[use_loader][1])#[1]: value of t_val, time period per real data
                            elif isinstance(out, dict):
                                loss += training_loss(**out, **sample,t_val=train_loaders[use_loader][1])
                    else:
                        if isinstance(out, torch.Tensor):
                            loss = training_loss(out.float(), **sample,t_val=train_loaders[use_loader][1])
                        elif isinstance(out, dict):
                            loss += training_loss(**out, **sample,t_val=train_loaders[use_loader][1])
                
                del out

                if regularizer:
                    loss += regularizer.loss
                
                loss.backward()

                optimizer.step()
                train_err += loss.item()
        
                with torch.no_grad():
                    avg_loss += loss.item()
                    if regularizer:
                        avg_lasso_loss += regularizer.loss

                if self.callbacks:
                    self.callbacks.on_batch_end()

            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(train_err)
            else:
                scheduler.step()

            epoch_train_time = default_timer() - t1            

            train_err /= print_normalize
            avg_loss  /= self.n_epochs
            if check_mem:
                myt.mmm(epoch)
            if epoch % self.log_test_interval == 0: 

                if self.callbacks:
                    self.callbacks.on_before_val(epoch=epoch, train_err=train_err, time=epoch_train_time, \
                                           avg_loss=avg_loss, avg_lasso_loss=avg_lasso_loss)
                

                for loader_name, loader in test_loaders.items():#'''#######evluation  here!!!!!!!!!'''
                    _ = self.evaluate(eval_losses, loader, log_prefix=loader_name,losstype=losstype)

                if self.callbacks:
                    self.callbacks.on_val_end()
            
            if self.callbacks:
                self.callbacks.on_epoch_end(epoch=epoch, train_err=train_err, avg_loss=avg_loss)
            if check_mem:
                myt.mmm(f'test{epoch}')
    def evaluate(self, loss_dict, data_loader,
                 log_prefix='',losstype='sum'):
        """Evaluates the model on a dictionary of losses
        
        Parameters
        ----------
        loss_dict : dict of functions 
          each function takes as input a tuple (prediction, ground_truth)
          and returns the corresponding loss
        data_loader : data_loader to evaluate on   changed to a dict of loader: { value:[loader, t_val]}
        log_prefix : str, default is ''
            if not '', used as prefix in output dictionary

        Returns
        -------
        errors : dict
            dict[f'{log_prefix}_{loss_name}] = loss for loss in loss_dict
        """

        if self.callbacks:
            self.callbacks.on_val_epoch_start(log_prefix=log_prefix, loss_dict = loss_dict, data_loader=data_loader)

        self.model.eval()

        errors = {f'{log_prefix}_{loss_name}':0 for loss_name in loss_dict.keys()}

        n_samples = 0
        with torch.no_grad():
            for idx, sample in enumerate(data_loader[0]):
                
                if self.callbacks:
                    self.callbacks.on_val_batch_start(idx=idx, sample=sample)
                
                y = sample['y']
                n_samples += y.size(0)

                # load everything from the batch onto self.device if 
                # no callback overrides default load to device
                
                if self.override_load_to_device:
                    self.callbacks.on_load_to_device(sample=sample)
                else:
                    for k,v in sample.items():
                        if hasattr(v, 'to'):
                            sample[k] = v.to(self.device)

                out = self.model(**sample)

                if self.callbacks:
                    self.callbacks.on_before_val_loss(out=out)
                
                for loss_name, loss in loss_dict.items():
                    if self.overrides_loss:
                        if isinstance(out, torch.Tensor):
                            val_loss = self.callbacks.compute_training_loss(out.float(), **sample,t_val=data_loader[1])
                        elif isinstance(out, dict):
                            val_loss = self.callbacks.compute_training_loss(**out, **sample,t_val=data_loader[1])
                    else:
                        if isinstance(out, torch.Tensor):
                            val_loss = loss(out, **sample,t_val=data_loader[1]).item()
                        elif isinstance(out, dict):
                            val_loss = loss(out, **sample,t_val=data_loader[1]).item()

                    errors[f'{log_prefix}_{loss_name}'] += val_loss

                if self.callbacks:
                    self.callbacks.on_val_batch_end()
        
        del y, out

        for key in errors.keys():
            if losstype=='sum':
                errors[key] /= n_samples
            elif losstype=='mean':
                errors[key]/=len(data_loader)
            else:
                print('error_key: in trainer.py, L327 eval loss')
        
        if self.callbacks:
            self.callbacks.on_val_epoch_end(errors=errors)

        return errors
