from functools import partial

import numpy as np
from torch_lr_finder import LRFinder
import optuna
from optuna.trial import TrialState
from optuna.samplers import TPESampler, GridSampler
from addict import Dict
import time
import datetime

from torchreid.utils import get_model_attr
from torchreid.optim import SAM


class WrappedLRFinder(LRFinder):
    def _train_batch(self, train_iter, accumulation_steps, non_blocking_transfer=True):
        self.model.train()
        total_loss = None  # for late initialization

        self.optimizer.zero_grad()
        for _ in range(accumulation_steps):
            inputs, labels = next(train_iter)
            inputs, labels = self._move_to_device(
                inputs, labels, non_blocking=non_blocking_transfer
            )
            # Forward pass
            loss = self.forward_pass(inputs, labels, accumulation_steps)

            if not isinstance(self.optimizer, SAM):
                self.optimizer.step()
            else:
                self.optimizer.first_step()
                loss = self.forward_pass(inputs, labels, accumulation_steps)
                self.optimizer.second_step()

            if total_loss is None:
                total_loss = loss
            else:
                total_loss += loss

        return total_loss.item()

    def forward_pass(self, inputs, targets, accumulation_steps):
        # Forward pass
        model_output = self.model(inputs)
        all_logits = model_output[0] if isinstance(model_output, (tuple, list)) else model_output
        loss = self.criterion(all_logits, targets)

        # Loss should be averaged in each step
        loss /= accumulation_steps

        # Backward pass
        loss.backward()
        return loss


class LrFinder:
    def __init__(self,
        engine,
        mode='fast_ai',
        epochs_warmup=2,
        max_lr=0.03,
        min_lr=4e-3,
        step=0.001,
        num_epochs=3,
        path_to_savefig='',
        seed = 5,
        stop_callback=None,
        smooth_f=0.01,
        n_trials=30,
        **kwargs) -> None:
        r"""A  pipeline for learning rate search.
        Args:
            mode (str, optional): mode for learning rate finder, "fast_ai", "grid_search", "TPE".
                Default is "fast_ai".
            max_lr (float): upper bound for leaning rate
            min_lr (float): lower bound for leaning rate
            step (float, optional): number of step for learning rate searching space. Default is 1e-3
            num_epochs (int, optional): number of epochs to train for each learning rate. Default is 3
            pretrained (bool): whether or not the model is pretrained
            path_to_savefig (str): if path given save plot loss/lr (only for fast_ai mode). Default: ''
        """
        self.engine = engine
        main_model_name = engine.get_model_names(None)[0]
        self.model = engine.models[main_model_name]
        self.optimizer = engine.optims[main_model_name]
        self.model_device = next(self.model.parameters()).device
        self.mode = mode
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.step = step
        self.n_trials = n_trials
        self.num_epochs = num_epochs
        # assert self.num_epochs > 3, "Number of epochs to find an optimal learning rate less than 3. It's pointless"
        self.path_to_savefig = path_to_savefig
        self.seed = seed
        self.stop_callback = stop_callback
        self.epochs_warmup = epochs_warmup
        self.enable_sam = engine.enable_sam
        self.smooth_f = smooth_f
        self.engine_cfg = Dict(min_lr=min_lr, max_lr=max_lr, mode=mode, step=step)
        search_space = np.arange(min_lr, max_lr, step)
        self.samplers = {'grid_search': GridSampler(search_space={'lr': search_space}),
                            'TPE': TPESampler(n_startup_trials=5, seed=True)}

    def process(self):
        print('=> Start learning rate search. Mode: {}'.format(self.mode))
        if self.mode == 'fast_ai':
           lr = self.fast_ai()
           return lr

        assert self.mode in ['grid_search', 'TPE']
        lr = self.optuna_optim()
        return lr

    def fast_ai(self):
        criterion = self.engine.main_losses[0]

        if self.epochs_warmup != 0:
            get_model_attr(self.model, 'to')(self.model_device)
            print("Warmup the model's weights for {} epochs".format(self.epochs_warmup))
            self.engine.run(max_epoch=self.epochs_warmup, lr_finder=self.engine_cfg, stop_callback=self.stop_callback)
            print("Finished warmuping the model. Continue to find learning rate:")

        # run lr finder
        num_iter = len(self.engine.train_loader)
        lr_finder = WrappedLRFinder(self.model, self.optimizer, criterion, device=self.model_device)
        lr_finder.range_test(self.engine.train_loader, start_lr=self.min_lr, end_lr=self.max_lr,
                                smooth_f=self.smooth_f, num_iter=num_iter, step_mode='exp')
        ax, optim_lr = lr_finder.plot(suggest_lr=True)
        # save plot if needed
        if self.path_to_savefig:
            fig = ax.get_figure()
            fig.savefig(self.path_to_savefig)

        # reset weights and optimizer state
        if self.epochs_warmup != 0:
            self.engine.restore_model()
        else:
            lr_finder.reset()

        return optim_lr

    def optuna_optim(self):
        def finish_process(study):
            pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
            complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

            print("Study statistics: ")
            print("  Number of finished trials: ", len(study.trials))
            print("  Number of pruned trials: ", len(pruned_trials))
            print("  Number of complete trials: ", len(complete_trials))

            print("Best trial:")
            trial = study.best_trial

            print("  Value: ", trial.value)

            print("  Params: ")
            for key, value in trial.params.items():
                print("    {}: {}".format(key, value))

            return trial.params['lr']

        study = optuna.create_study(study_name='classification task', direction="maximize", sampler=self.samplers[self.mode])
        objective_partial = partial(self.engine.run, max_epoch=self.num_epochs, lr_finder=self.engine_cfg, start_eval=0, eval_freq=1,
                                stop_callback=self.stop_callback)
        try:
            start_time = time.time()
            study.optimize(objective_partial, n_trials=self.n_trials, timeout=None)
            elapsed = round(time.time() - start_time)
            print(f"--- learning rate estimation finished with elapsed time: {datetime.timedelta(seconds=elapsed)} ---")

        except KeyboardInterrupt:
            return finish_process(study)

        else:
            return finish_process(study)
