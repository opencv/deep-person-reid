# Copyright (c) 2018-2021 Kaiyang Zhou
# SPDX-License-Identifier: MIT
#
# Copyright (C) 2020-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from __future__ import absolute_import, print_function
import math

from torch import optim
from torch.optim.lr_scheduler import _LRScheduler

AVAI_SCH = {'single_step', 'multi_step', 'cosine', 'warmup', 'cosine_cycle', 'reduce_on_plateau', 'onecycle'}

def build_lr_scheduler(optimizer, lr_scheduler, base_scheduler, **kwargs):
    if lr_scheduler == 'warmup':
        base_scheduler = _build_scheduler(optimizer=optimizer, lr_scheduler=base_scheduler, base_scheduler=None, **kwargs)
        scheduler = _build_scheduler(optimizer=optimizer, lr_scheduler=lr_scheduler, base_scheduler=base_scheduler, **kwargs)
    else:
        scheduler = _build_scheduler(optimizer=optimizer, lr_scheduler=lr_scheduler, base_scheduler=None, **kwargs)

    return scheduler

def _build_scheduler(optimizer,
                     num_iter,
                     lr_scheduler='single_step',
                     base_scheduler=None,
                     stepsize=1,
                     gamma=0.1,
                     max_epoch=1,
                     warmup=10,
                     multiplier=10,
                     first_cycle_steps=10,
                     cycle_mult=1.,
                     min_lr=1e-4,
                     max_lr=0.1,
                     patience=5,
                     lr_decay_factor=100,
                     pct_start=0.3):

    init_learning_rate = [param_group['lr'] for param_group in optimizer.param_groups]
    if lr_scheduler not in AVAI_SCH:
        raise ValueError('Unsupported scheduler: {}. Must be one of {}'.format(lr_scheduler, AVAI_SCH))

    if isinstance(base_scheduler, WarmupScheduler):
        raise ValueError(
            'Invalid base scheduler. WarmupScheduler cannot be the base one'
        )

    if lr_scheduler == 'single_step':
        if isinstance(stepsize, list):
            stepsize = stepsize[-1]

        if not isinstance(stepsize, int):
            raise TypeError(
                'For single_step lr_scheduler, stepsize must '
                'be an integer, but got {}'.format(type(stepsize))
            )

        scheduler = optim.lr_scheduler.StepLR(
            optimizer, step_size=stepsize, gamma=gamma
        )

    elif lr_scheduler == 'multi_step':
        if not isinstance(stepsize, list):
            raise TypeError(
                'For multi_step lr_scheduler, stepsize must '
                'be a list, but got {}'.format(type(stepsize))
            )

        scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=stepsize, gamma=gamma
        )

    elif lr_scheduler == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, float(max_epoch)
        )

    elif lr_scheduler == 'warmup':
        if base_scheduler is None:
            raise ValueError("Base scheduler is not defined. Please, add it to the configuration file.")
        scheduler = WarmupScheduler(optimizer, multiplier=multiplier, total_epoch=warmup, after_scheduler=base_scheduler)

    elif lr_scheduler == 'cosine_cycle':
        scheduler = CosineAnnealingCycleRestart(optimizer, first_cycle_steps=first_cycle_steps, cycle_mult=cycle_mult,
        max_lr=max_lr, min_lr=min_lr, warmup_steps=warmup, gamma=gamma)

    elif lr_scheduler == 'onecycle':
        scheduler = OneCycleLR(optimizer, max_lr=init_learning_rate, steps_per_epoch=num_iter,
                               epochs=int(max_epoch), pct_start=pct_start, final_div_factor=lr_decay_factor,
                               div_factor=100)

    elif lr_scheduler == 'reduce_on_plateau':
        lb_lr = [lr / lr_decay_factor for lr in init_learning_rate]
        epoch_treshold = max(int(max_epoch * 0.75) - warmup, 1) # 75% of the training - warmup epochs
        scheduler = ReduceLROnPlateauV2(optimizer, epoch_treshold, factor=gamma, patience=patience,
                                        threshold=2e-4, verbose=True, min_lr=lb_lr )
    else:
        raise ValueError('Unknown scheduler: {}'.format(lr_scheduler))

    return scheduler


class CosineAnnealingCycleRestart(_LRScheduler):
    """
        optimizer (Optimizer): Wrapped optimizer.
        first_cycle_steps (int): First cycle step size.
        cycle_mult(float): Cycle steps magnification. Default: -1.
        max_lr(float): First cycle's max learning rate. Default: 0.1.
        min_lr(float): Min learning rate. Default: 0.001.
        warmup_steps(int): Linear warmup step size. Default: 0.
        gamma(float): Decrease rate of max learning rate by cycle. Default: 1.
        last_epoch (int): The index of last epoch. Default: -1.
    """

    def __init__(self,
                 optimizer : optim.Optimizer,
                 first_cycle_steps : int,
                 cycle_mult : float = 1.,
                 max_lr : float = 0.1,
                 min_lr : list = 0.001,
                 warmup_steps : int = 0,
                 gamma : float = 1.,
                 last_epoch : int = -1
        ):
        assert warmup_steps < first_cycle_steps

        self.first_cycle_steps = first_cycle_steps # first cycle step size
        self.cycle_mult = cycle_mult # cycle steps magnification
        self.base_max_lr = max_lr # first max learning rate
        self.max_lr = max_lr # max learning rate in the current cycle
        self.min_lr = min_lr # min learning rate
        self.warmup_steps = warmup_steps # warmup step size
        self.gamma = gamma # decrease rate of max learning rate by cycle

        self.cur_cycle_steps = first_cycle_steps # first cycle step size
        self.cycle = 0 # cycle count
        self.step_in_cycle = last_epoch # step size of the current cycle

        super().__init__(optimizer, last_epoch)

        # set learning rate min_lr
        self.init_lr()

    def init_lr(self):
        self.base_lrs = []
        for lr, param_group in zip(self.min_lr, self.optimizer.param_groups):
            param_group['lr'] = lr
            self.base_lrs.append(lr)

    def get_lr(self):
        if self.step_in_cycle == -1:
            return self.base_lrs
        elif self.step_in_cycle < self.warmup_steps:
            return [(self.max_lr - base_lr)*self.step_in_cycle / self.warmup_steps + base_lr for base_lr in self.base_lrs]
        else:
            return [base_lr + (self.max_lr - base_lr) \
                    * (1 + math.cos(math.pi * (self.step_in_cycle-self.warmup_steps) \
                                    / (self.cur_cycle_steps - self.warmup_steps))) / 2
                    for base_lr in self.base_lrs]

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
            self.step_in_cycle = self.step_in_cycle + 1
            if self.step_in_cycle >= self.cur_cycle_steps:
                self.cycle += 1
                self.step_in_cycle = self.step_in_cycle - self.cur_cycle_steps
                self.cur_cycle_steps = int((self.cur_cycle_steps - self.warmup_steps) * self.cycle_mult) + self.warmup_steps
        else:
            if epoch >= self.first_cycle_steps:
                if self.cycle_mult == 1.:
                    self.step_in_cycle = epoch % self.first_cycle_steps
                    self.cycle = epoch // self.first_cycle_steps
                else:
                    n = int(math.log((epoch / self.first_cycle_steps * (self.cycle_mult - 1) + 1), self.cycle_mult))
                    self.cycle = n
                    self.step_in_cycle = epoch - int(self.first_cycle_steps * (self.cycle_mult ** n - 1) / (self.cycle_mult - 1))
                    self.cur_cycle_steps = self.first_cycle_steps * self.cycle_mult ** (n)
            else:
                self.cur_cycle_steps = self.first_cycle_steps
                self.step_in_cycle = epoch

        self.max_lr = self.base_max_lr * (self.gamma**self.cycle)
        self.last_epoch = math.floor(epoch)
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr

class WarmupScheduler(_LRScheduler):
    def __init__(self, optimizer, multiplier, total_epoch, after_scheduler=None):
        self.multiplier = multiplier
        self.total_epoch = total_epoch
        self.after_scheduler = after_scheduler
        self.num_bad_epochs = 0
        self.warmup_finished = False
        super().__init__(optimizer)
        # scale down initial learning rate
        self.init_lr()

    def get_lr(self):
        if self.last_epoch > self.total_epoch:
            if self.after_scheduler:
                if not self.warmup_finished:
                    self.after_scheduler.base_lrs = [base_lr * self.multiplier for base_lr in self.base_lrs]
                    self.warmup_finished = True
            return [base_lr * self.multiplier for base_lr in self.base_lrs]

        res = [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_epoch + 1.) for base_lr in self.base_lrs]
        return res

    def init_lr(self):
        self.base_lrs = []
        for param_group in self.optimizer.param_groups:
            param_group['lr'] /= self.multiplier
            self.base_lrs.append(param_group['lr'])

    def step(self, epoch=None, metrics=None):
        if self.warmup_finished and self.after_scheduler:
            if isinstance(self.after_scheduler, ReduceLROnPlateauV2):
                if epoch is None:
                    self.after_scheduler.step(metrics=metrics, epoch=None)
                else:
                    self.after_scheduler.step(metrics=metrics, epoch=epoch - self.total_epoch)
                self.num_bad_epochs = self.after_scheduler.num_bad_epochs
            else:
                if epoch is None:
                    self.after_scheduler.step(None)
                else:
                    self.after_scheduler.step(epoch - self.total_epoch)
        else:
            return super().step(epoch)

    def load_state_dict(self, state_dict):
        """Loads the schedulers state.
        """
        self.warmup_finished = state_dict['warmup_finished']
        self.total_epoch = state_dict['total_epoch']
        self.last_epoch = state_dict['last_epoch']
        self.multiplier = state_dict['multiplier']

        self.after_scheduler.load_state_dict(state_dict)

class ReduceLROnPlateauV2(optim.lr_scheduler.ReduceLROnPlateau):
    def __init__(self,
                 optimizer: optim.Optimizer,
                 epoch_treshold: int,
                 **kwargs) -> None:

        super().__init__(optimizer, **kwargs)
        self.epoch_treshold = epoch_treshold
        self.init_lr = [group['lr'] for group in self.optimizer.param_groups]

    def step(self, metrics, epoch=None):
        super().step(metrics, epoch=epoch)

        lr_reduced_flag = self.is_reduced()
        # if there is no learning rate decay for more than {self.epoch_treshold} epochs
        # we force to do that
        if self.last_epoch >= self.epoch_treshold and not lr_reduced_flag:
            print("Force learning rate decaying...")
            self._reduce_lr(self.last_epoch)
            self.cooldown_counter = self.cooldown
            self.num_bad_epochs = 0
            self._last_lr = [group['lr'] for group in self.optimizer.param_groups]

    def is_reduced(self):
        if any([current_lr < init_lr for  init_lr, current_lr in zip(self.init_lr, self._last_lr)]):
            return True
        return False


class OneCycleLR(optim.lr_scheduler.OneCycleLR):
    @property
    def warmup_finished(self):
        print(self._schedule_phases[0]['end_step'], self.last_epoch)
        return self.last_epoch >= self._schedule_phases[0]['end_step']