from __future__ import absolute_import, division, print_function
import os
import os.path as osp
import gdown

import pickle
import warnings
from collections import OrderedDict
from functools import partial
from pprint import pformat
from copy import deepcopy

import torch
import torch.nn as nn
from torch.nn.modules import module

from .tools import mkdir_if_missing, check_isfile
from torchreid.models import TimmModelsWrapper

__all__ = [
    'save_checkpoint', 'load_checkpoint', 'resume_from_checkpoint',
    'open_all_layers', 'open_specified_layers', 'count_num_param',
    'load_pretrained_weights', 'ModelEmaV2'
]


def params_to_device(param, device):
    def tensor_to_device(param, device):
        param.data = param.data.to(device)
        if param._grad is not None:
            param._grad.data = param._grad.data.to(device)

    if isinstance(param, torch.Tensor):
        tensor_to_device(param, device)
    elif isinstance(param, dict):
        for subparam in param.values():
            tensor_to_device(subparam, device)

def optimizer_to(optim, device):
    for param in optim.state.values():
        params_to_device(param, device)

def scheduler_to(sched, device):
    for param in sched.__dict__.values():
        params_to_device(param, device)

def save_checkpoint(
    state, save_dir, is_best=False, remove_module_from_keys=False, name='model'
):
    r"""Saves checkpoint.

    Args:
        state (dict): dictionary.
        save_dir (str): directory to save checkpoint.
        is_best (bool, optional): if True, this checkpoint will be copied and named
            ``model-best.pth.tar``. Default is False.
        remove_module_from_keys (bool, optional): whether to remove "module."
            from layer names. Default is False.

    Examples::
        >>> state = {
        >>>     'state_dict': model.state_dict(),
        >>>     'epoch': 10,
        >>>     'rank1': 0.5,
        >>>     'optimizer': optimizer.state_dict()
        >>> }
        >>> save_checkpoint(state, 'log/my_model')
    """
    mkdir_if_missing(save_dir)
    if remove_module_from_keys:
        # remove 'module.' in state_dict's keys
        state_dict = state['state_dict']
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            if k.startswith('module.'):
                k = k[7:]
            new_state_dict[k] = v
        state['state_dict'] = new_state_dict
    # save
    epoch = state['epoch']
    fpath = osp.join(save_dir, f'{name}.pth.tar-' + str(epoch))
    torch.save(state, fpath)
    print('Checkpoint saved to "{}"'.format(fpath))
    if is_best:
        best_link_path = osp.join(osp.dirname(fpath), f'{name}-best.pth.tar')
        if osp.lexists(best_link_path):
            os.remove(best_link_path)
        basename_fpath = osp.basename(fpath)
        print(f'Creating best link {basename_fpath} -> {best_link_path}')
        os.symlink(basename_fpath, best_link_path)
    return fpath


def download_weights(url, chkpt_name='model_weights'):
    """ Download model weights from given url """
    def _get_torch_home():
        ENV_TORCH_HOME = 'TORCH_HOME'
        ENV_XDG_CACHE_HOME = 'XDG_CACHE_HOME'
        DEFAULT_CACHE_DIR = '~/.cache'
        torch_home = os.path.expanduser(
            os.getenv(
                ENV_TORCH_HOME,
                os.path.join(
                    os.getenv(ENV_XDG_CACHE_HOME, DEFAULT_CACHE_DIR), 'torch'
                )
            )
        )
        return torch_home

    torch_home = _get_torch_home()
    model_dir = os.path.join(torch_home, 'checkpoints')
    os.makedirs(model_dir, exist_ok=True)

    filename = chkpt_name + '.pth'
    cached_file = os.path.join(model_dir, filename)
    if not os.path.exists(cached_file):
        try:
            gdown.download(url, cached_file)
        except Exception as e:
            print(f"ERROR:: error occurred while \
                    downloading file")
            raise e
    try:
        torch.load(cached_file)
    except Exception as e:
        print(f"ERROR:: error occurred while opening \
                file with model weights")
        raise e
    else:
        print("SUCCESS:: Model`s weights download completed successfully\n")
    return cached_file


def load_checkpoint(fpath, map_location=''):
    r"""Loads checkpoint.

    ``UnicodeDecodeError`` can be well handled, which means
    python2-saved files can be read from python3.

    Args:
        fpath (str): path to checkpoint.

    Returns:
        dict

    Examples::
        >>> from torchreid.utils import load_checkpoint
        >>> fpath = 'log/my_model/model.pth.tar-10'
        >>> checkpoint = load_checkpoint(fpath)
    """
    if fpath is None:
        raise ValueError('File path is None')
    if not osp.exists(fpath):
        raise FileNotFoundError('File is not found at "{}"'.format(fpath))
    if not map_location:
        map_location = None if torch.cuda.is_available() else 'cpu'
    try:
        checkpoint = torch.load(fpath, map_location=map_location)
    except UnicodeDecodeError:
        pickle.load = partial(pickle.load, encoding="latin1")
        pickle.Unpickler = partial(pickle.Unpickler, encoding="latin1")
        checkpoint = torch.load(
            fpath, pickle_module=pickle, map_location=map_location
        )
    except Exception:
        print('Unable to load checkpoint from "{}"'.format(fpath))
        raise
    return checkpoint


def resume_from_checkpoint(fpath, model, optimizer=None, scheduler=None, device='cpu'):
    r"""Resumes training from a checkpoint.

    This will load (1) model weights and (2) ``state_dict``
    of optimizer if ``optimizer`` is not None.

    Args:
        fpath (str): path to checkpoint.
        model (nn.Module): model.
        optimizer (Optimizer, optional): an Optimizer.
        scheduler (LRScheduler, optional): an LRScheduler.

    Returns:
        int: start_epoch.

    Examples::
        >>> from torchreid.utils import resume_from_checkpoint
        >>> fpath = 'log/my_model/model.pth.tar-10'
        >>> start_epoch = resume_from_checkpoint(
        >>>     fpath, model, optimizer, scheduler
        >>> )
    """
    is_file = check_isfile(fpath)
    if not is_file:
        # Then link is presented or something different
        # that will be checked and processed in download function
        chkpt_name = model.__class__.__name__ + "_resume"
        fpath = download_weights(fpath, chkpt_name=chkpt_name)

    print('Loading checkpoint from "{}"'.format(fpath))
    checkpoint = load_checkpoint(fpath)
    if 'state_dict' in checkpoint:
        load_pretrained_weights(model, pretrained_dict=checkpoint['state_dict'])
    else:
        load_pretrained_weights(model, pretrained_dict=checkpoint)
    print('Loaded model weights')
    if optimizer is not None and 'optimizer' in checkpoint.keys():
        optimizer.load_state_dict(checkpoint['optimizer'])
        optimizer_to(optimizer, device)
        print('Loaded optimizer')
    if scheduler is not None and 'scheduler' in checkpoint.keys():
        scheduler.load_state_dict(checkpoint['scheduler'])
        scheduler_to(scheduler, device)
        print('Loaded scheduler')
    if 'epoch' in checkpoint:
        start_epoch = checkpoint['epoch']
    else:
        start_epoch = 0
    print('Last epoch = {}'.format(start_epoch))
    if 'rank1' in checkpoint.keys():
        print('Last rank1 = {:.1%}'.format(checkpoint['rank1']))
    return start_epoch


def adjust_learning_rate(
    optimizer,
    base_lr,
    epoch,
    stepsize=20,
    gamma=0.1,
    linear_decay=False,
    final_lr=0,
    max_epoch=100
):
    r"""Adjusts learning rate.

    Deprecated.
    """
    if linear_decay:
        # linearly decay learning rate from base_lr to final_lr
        frac_done = epoch / max_epoch
        lr = frac_done*final_lr + (1.-frac_done) * base_lr
    else:
        # decay learning rate by gamma for every stepsize
        lr = base_lr * (gamma**(epoch // stepsize))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def set_bn_to_eval(m):
    r"""Sets BatchNorm layers to eval mode."""
    # 1. no update for running mean and var
    # 2. scale and shift parameters are still trainable
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.eval()


def open_all_layers(model):
    r"""Opens all layers in model for training.

    Examples::
        >>> from torchreid.utils import open_all_layers
        >>> open_all_layers(model)
    """
    model.train()
    for p in model.parameters():
        if p.dtype in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64, torch.bool):
            # only Tensors of floating point dtype can require gradients
            continue
        p.requires_grad = True


def open_specified_layers(model, open_layers, strict=True):
    r"""Opens specified layers in model for training while keeping
    other layers frozen.

    Args:
        model (nn.Module): neural net model.
        open_layers (str or list): layers open for training.

    Examples::
        >>> from torchreid.utils import open_specified_layers
        >>> # Only model.classifier will be updated.
        >>> open_layers = 'classifier'
        >>> open_specified_layers(model, open_layers)
        >>> # Only model.fc and model.classifier will be updated.
        >>> open_layers = ['fc', 'classifier']
        >>> open_specified_layers(model, open_layers)
    """
    if isinstance(model, nn.DataParallel):
        model = model.module

    if isinstance(open_layers, str):
        open_layers = [open_layers]

    if strict:
        for layer in open_layers:
            if not hasattr(model, layer):
                raise ValueError('"{}" is not an attribute of the model, please provide the correct name'.format(layer))

    for name, module in model.named_children():
        if name in open_layers:
            module.train()
            for p in module.parameters():
                p.requires_grad = True
        else:
            module.eval()
            for p in module.parameters():
                p.requires_grad = False


def count_num_param(model):
    r"""Counts number of parameters in a model while ignoring ``self.classifier``.

    Args:
        model (nn.Module): network model.

    Examples::
        >>> from torchreid.utils import count_num_param
        >>> model_size = count_num_param(model)

    .. warning::

        This method is deprecated in favor of
        ``torchreid.utils.compute_model_complexity``.
    """
    warnings.warn(
        'This method is deprecated and will be removed in the future.'
    )

    num_param = sum(p.numel() for p in model.parameters())

    if isinstance(model, nn.DataParallel):
        model = model.module

    if hasattr(model,
               'classifier') and isinstance(model.classifier, nn.Module):
        # we ignore the classifier because it is unused at test time
        num_param -= sum(p.numel() for p in model.classifier.parameters())

    return num_param


def _print_loading_weights_inconsistencies(discarded_layers, unmatched_layers):
    if discarded_layers:
        print(
            '** The following layers are discarded '
            'due to unmatched keys or layer size: {}'.
            format(pformat(discarded_layers))
        )
    if unmatched_layers:
        print(
            '** The following layers were not loaded from checkpoint: {}'.
            format(pformat(unmatched_layers))
        )


def load_pretrained_weights(model, file_path='', chkpt_name='model_weights', pretrained_dict=None):
    r"""Loads pretrianed weights to model.
    Features::
        - Incompatible layers (unmatched in name or size) will be ignored.
        - Can automatically deal with keys containing "module." and other prefixes.
        - Can download weights from link
        - Can use pretrained dict directly instead of file path/link if given
    Args:
        model (nn.Module): network model.
        file_path (str): path or link to pretrained weights.
        pretrained_dict (str): path or link to pretrained weights.
    Examples::
        >>> from torchreid.utils import load_pretrained_weights
        >>> file_path = 'log/my_model/model-best.pth.tar'
        >>> load_pretrained_weights(model, file_path)
    """
    def _add_prefix(key, prefix):
        key = prefix + "." + key
        return key

    def _remove_prefix(key, prefix):
        prefix = prefix + '.'
        if key.startswith(prefix):
            key = key[len(prefix):]
        return key

    is_file = check_isfile(file_path)
    pretraining_timm_model = model.pretrained and isinstance(model, TimmModelsWrapper)
    if not is_file and not pretrained_dict:
        # Then link is presented or something different
        # that will be checked and processed in download function
        file_path = download_weights(file_path, chkpt_name=chkpt_name)

    checkpoint = (load_checkpoint(file_path)
                    if not pretrained_dict
                    else pretrained_dict)

    if 'classes_map' in checkpoint:
        model.classification_classes = checkpoint['classes_map']
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    elif 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint

    model_dict = model.state_dict()
    new_state_dict = OrderedDict()
    matched_layers, discarded_layers = [], []

    for k, v in state_dict.items():
        # discard known prefixes: 'nncf_module.' from NNCF, 'module.' from DataParallel
        k = _remove_prefix(k, 'nncf_module')
        k = _remove_prefix(k, 'module')
        if pretraining_timm_model:
            k = _add_prefix(k, 'model')

        if k in model_dict and model_dict[k].size() == v.size():
            new_state_dict[k] = v
            matched_layers.append(k)
        else:
            discarded_layers.append(k)

    model_dict.update(new_state_dict)
    model.load_state_dict(model_dict)

    message = file_path if file_path else "pretrained dict"
    unmatched_layers = sorted(set(model_dict.keys()) - set(new_state_dict))
    if len(matched_layers) == 0:
        print(
            'The pretrained weights "{}" cannot be loaded, '
            'please check the key names manually'.format(message)
        )
        _print_loading_weights_inconsistencies(discarded_layers, unmatched_layers)

        raise RuntimeError(f'The pretrained weights {message} cannot be loaded')
    else:
        print(
            'Successfully loaded pretrained weights from "{}"'.
            format(message)
        )
        _print_loading_weights_inconsistencies(discarded_layers, unmatched_layers)


class ModelEmaV2(nn.Module):
    """ Model Exponential Moving Average V2
    Keep a moving average of everything in the model state_dict (parameters and buffers).
    V2 of this module is simpler, it does not match params/buffers based on name but simply
    iterates in order. It works with torchscript (JIT of full model).
    This is intended to allow functionality like
    https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage
    A smoothed version of the weights is necessary for some training schemes to perform well.
    E.g. Google's hyper-params for training MNASNet, MobileNet-V3, EfficientNet, etc that use
    RMSprop with a short 2.4-3 epoch decay period and slow LR decay rate of .96-.99 requires EMA
    smoothing of weights to match results. Pay attention to the decay constant you are using
    relative to your update count per epoch.
    To keep EMA from using GPU resources, set device='cpu'. This will save a bit of memory but
    disable validation of the EMA weights. Validation will have to be done manually in a separate
    process, or after the training stops converging.
    This class is sensitive where it is initialized in the sequence of model init,
    GPU assignment and distributed training wrappers.
    """
    def __init__(self, model, decay=0.9999, device=None):
        super(ModelEmaV2, self).__init__()
        # make a copy of the model for accumulating moving average of weights
        self.module = deepcopy(model)
        self.module.eval()
        self.decay = decay
        self.device = device  # perform ema on different device from model if set
        if self.device is not None:
            self.module.to(device=device)

    def _update(self, model, update_fn):
        with torch.no_grad():
            for ema_v, model_v in zip(self.module.state_dict().values(), model.state_dict().values()):
                if self.device is not None:
                    model_v = model_v.to(device=self.device)
                ema_v.copy_(update_fn(ema_v, model_v))

    def update(self, model):
        self._update(model, update_fn=lambda e, m: self.decay * e + (1. - self.decay) * m)

    def set(self, model):
        self._update(model, update_fn=lambda e, m: m)
