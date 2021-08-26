import os
import yaml

from yacs.config import CfgNode as CN


def get_default_config():

    cfg = CN()

    # special slot for inheritance implementation
    #  -- see the function merge_from_files_with_base below
    cfg._base_ = ''

    # lr finder
    cfg.lr_finder = CN()
    cfg.lr_finder.enable = False
    cfg.lr_finder.mode = 'fast_ai'
    cfg.lr_finder.max_lr = 0.03
    cfg.lr_finder.min_lr = 0.004
    cfg.lr_finder.step = 0.001
    cfg.lr_finder.num_epochs = 3
    cfg.lr_finder.epochs_warmup = 2
    cfg.lr_finder.stop_after = False
    cfg.lr_finder.path_to_savefig = ''
    cfg.lr_finder.smooth_f = 0.01
    cfg.lr_finder.n_trials = 100

    # model
    cfg.model = CN()
    cfg.model.name = 'resnet50'
    cfg.model.pretrained = False
    cfg.model.download_weights = True
    cfg.model.load_weights = '' # path to snapshot to load weights
    cfg.model.save_chkpt = True
    cfg.model.resume = '' # path to checkpoint for resume training
    cfg.model.dropout_backbone = CN()
    cfg.model.dropout_backbone.p = 0.0
    cfg.model.dropout_backbone.mu = 0.1
    cfg.model.dropout_backbone.sigma = 0.03
    cfg.model.dropout_backbone.kernel = 3
    cfg.model.dropout_backbone.temperature = 0.2
    cfg.model.dropout_backbone.dist = 'none'
    cfg.model.dropout_cls = CN()
    cfg.model.dropout_cls.p = 0.0
    cfg.model.dropout_cls.mu = 0.1
    cfg.model.dropout_cls.sigma = 0.03
    cfg.model.dropout_cls.kernel = 3
    cfg.model.dropout_cls.temperature = 0.2
    cfg.model.dropout_cls.dist = 'none'
    cfg.model.feature_dim = 512  # embedding size
    cfg.model.bn_eval = False
    cfg.model.bn_frozen = False
    cfg.model.enable_attentions = False
    cfg.model.pooling_type = 'avg'
    cfg.model.IN_first = False
    cfg.model.IN_conv1 = False
    cfg.model.extra_blocks = False
    cfg.model.lct_gate = False
    cfg.model.fpn = CN()
    cfg.model.fpn.enable = True
    cfg.model.fpn.dim = 256
    cfg.model.fpn.process = 'concatenation'
    cfg.model.type = 'reid'
    cfg.model.self_challenging_cfg = CN()
    cfg.model.self_challenging_cfg.enable = False
    cfg.model.self_challenging_cfg.drop_p = 0.33
    cfg.model.self_challenging_cfg.drop_batch_p = 0.33

    # mutual learning, auxiliary model
    cfg.mutual_learning = CN()
    cfg.mutual_learning.aux_configs = []

    # data
    cfg.data = CN()
    cfg.data.root = 'reid-data'
    cfg.data.sources = ['market1501']
    cfg.data.targets = ['market1501']
    cfg.data.workers = 4  # number of data loading workers
    cfg.data.split_id = 0  # Split index
    cfg.data.height = 256  # image height
    cfg.data.width = 128  # image width
    cfg.data.enable_masks = False
    cfg.data.combineall = False  # combine train, query and gallery for training
    cfg.data.norm_mean = [0.485, 0.456, 0.406]  # default is imagenet mean
    cfg.data.norm_std = [0.229, 0.224, 0.225]  # default is imagenet std
    cfg.data.save_dir = 'log'  # path to save log
    cfg.data.tb_log_dir = ''  # path to save tensorboard log. If empty, log will be saved to data.save_dir
    cfg.data.min_samples_per_id = 1
    cfg.data.num_sampled_packages = 1

    # specific datasets
    cfg.market1501 = CN()
    cfg.market1501.use_500k_distractors = False  # add 500k distractors to the gallery set for market1501
    cfg.cuhk03 = CN()
    cfg.cuhk03.labeled_images = False  # use labeled images, if False, use detected images
    cfg.cuhk03.classic_split = False  # use classic split by Li et al. CVPR14
    cfg.cuhk03.use_metric_cuhk03 = False  # use cuhk03's metric for evaluation

    # custom_datasets
    cfg.custom_datasets = CN() # this node contains information about custom classification datasets
    cfg.custom_datasets.roots = [] # a list of root folders in case of ImagesFolder fromat or list of annotation files with paths relative to the list's parent folder
    cfg.custom_datasets.types = [] # a list of types (classification or classification_image_folder)
    cfg.custom_datasets.names = [] # aliases for custom datasets that can be used in the data section. Should be unique

    # sampler
    cfg.sampler = CN()
    cfg.sampler.train_sampler = 'RandomSampler'
    cfg.sampler.batch_num_instances = 4  # number of instances per identity for RandomIdentitySampler
    cfg.sampler.epoch_num_instances = -1
    cfg.sampler.fill_instances = False

    # train
    cfg.train = CN()
    cfg.train.optim = 'adam'
    cfg.train.base_optim = 'sgd'
    cfg.train.lr = 0.0003
    cfg.train.weight_decay = 5e-4
    cfg.train.max_epoch = 60
    cfg.train.start_epoch = 0
    cfg.train.batch_size = 32
    cfg.train.early_stoping = False # switch on exit on metric plataeu method
    cfg.train.train_patience = 10 # define how much epochs to wait after scheduler process
    cfg.train.open_layers = ['classifier']  # layers for training while keeping others frozen
    cfg.train.staged_lr = False  # set different lr to different layers
    cfg.train.new_layers = ['classifier']  # newly added layers with default lr
    cfg.train.base_lr_mult = 0.1  # learning rate multiplier for base layers
    cfg.train.lr_scheduler = 'single_step'
    cfg.train.target_metric = 'train_loss' # define which metric to use with reduce_on_plateau scheduler. Two possible variants are available: 'test_acc' and 'train_loss'
    cfg.train.base_scheduler = ''
    cfg.train.stepsize = [20]  # stepsize to decay learning rate
    cfg.train.gamma = 0.1  # learning rate decay multiplier
    cfg.train.first_cycle_steps = 5
    cfg.train.cycle_mult = 1.
    cfg.train.min_lr = 1e-5
    cfg.train.max_lr = 0.1
    cfg.train.lr_decay_factor = 100
    cfg.train.pct_start = 0.3
    cfg.train.fixbase_epoch = 0
    cfg.train.nbd = False
    cfg.train.patience = 5 # define how much epochs to wait for reduce on plateau
    cfg.train.multiplier = 10
    cfg.train.print_freq = 20  # print frequency
    cfg.train.seed = 5  # random seed
    cfg.train.deterministic = False # define to use cuda.deterministic
    cfg.train.warmup = 1  # After fixbase_epoch
    cfg.train.clip_grad = 0.
    cfg.train.ema = CN()
    cfg.train.ema.enable = False
    cfg.train.ema.ema_decay = 0.9999
    cfg.train.sam = CN()
    cfg.train.sam.rho = 0.05


    # optimizer
    cfg.sgd = CN()
    cfg.sgd.momentum = 0.9  # momentum factor for sgd and rmsprop
    cfg.sgd.dampening = 0.  # dampening for momentum
    cfg.sgd.nesterov = False  # Nesterov momentum
    cfg.rmsprop = CN()
    cfg.rmsprop.alpha = 0.99  # smoothing constant
    cfg.adam = CN()
    cfg.adam.beta1 = 0.9  # exponential decay rate for first moment
    cfg.adam.beta2 = 0.999  # exponential decay rate for second moment

    # loss
    cfg.loss = CN()
    cfg.loss.name = 'softmax'
    cfg.loss.softmax = CN()
    cfg.loss.softmax.label_smooth = 0.  # use label smoothing regularizer
    cfg.loss.softmax.margin_type = 'cos'
    cfg.loss.softmax.augmentations = CN()
    cfg.loss.softmax.augmentations.aug_type = '' # use advanced augmentations like fmix, cutmix and mixup
    cfg.loss.softmax.augmentations.alpha = 1.0
    cfg.loss.softmax.augmentations.aug_prob = 1.0
    cfg.loss.softmax.augmentations.fmix = CN()
    cfg.loss.softmax.augmentations.fmix.decay_power = 3
    cfg.loss.softmax.augmentations.fmix.max_soft = 0.0
    cfg.loss.softmax.augmentations.fmix.reformulate = False
    cfg.loss.softmax.conf_penalty = 0.0
    cfg.loss.softmax.pr_product = False
    cfg.loss.softmax.m = 0.35
    cfg.loss.softmax.s = 30.0
    cfg.loss.softmax.compute_s = False
    cfg.loss.softmax.end_s = -1.0
    cfg.loss.softmax.duration_s = -1
    cfg.loss.softmax.skip_steps_s = -1
    cfg.loss.softmax.adaptive_margins = False
    cfg.loss.softmax.class_weighting = False
    cfg.loss.softmax.base_num_classes = -1
    cfg.loss.softmax.symmetric_ce = False
    cfg.loss.asl = CN()
    cfg.loss.asl.gamma_pos = 0.
    cfg.loss.asl.gamma_neg = 0.
    cfg.loss.asl.p_m = 0.05
    cfg.loss.am_binary = CN()
    cfg.loss.am_binary.sym_adjustment = False
    cfg.loss.am_binary.auto_balance = False
    cfg.loss.am_binary.amb_k = 0.8
    cfg.loss.am_binary.amb_t = 1.
    cfg.loss.triplet = CN()
    cfg.loss.triplet.margin = 0.3  # distance margin
    cfg.loss.triplet.weight_t = 1.  # weight to balance hard triplet loss
    cfg.loss.triplet.weight_x = 0.  # weight to balance cross entropy loss

    # mixing loss
    cfg.mixing_loss = CN()
    cfg.mixing_loss.enable = False
    cfg.mixing_loss.weight = 1.0

    # metric_losses
    cfg.metric_losses = CN()
    cfg.metric_losses.enable = False
    cfg.metric_losses.center_coeff = 0.0
    cfg.metric_losses.triplet_coeff = 0.0
    cfg.metric_losses.local_push_coeff = 1.0
    cfg.metric_losses.center_margin = 0.1
    cfg.metric_losses.triplet_margin = 0.35
    cfg.metric_losses.local_push_margin = 0.1
    cfg.metric_losses.smart_margin = True
    cfg.metric_losses.triplet = 'semihard'
    cfg.metric_losses.loss_balancing = True
    cfg.metric_losses.centers_lr = 0.5
    cfg.metric_losses.balancing_lr = 0.01

    # attribute loss
    cfg.attr_loss = CN()
    cfg.attr_loss.names = ['color', 'type']
    cfg.attr_loss.num_classes = [-1, -1]
    cfg.attr_loss.label_smooth = False
    cfg.attr_loss.conf_penalty = 0.0
    cfg.attr_loss.pr_product = False
    cfg.attr_loss.m = 0.35
    cfg.attr_loss.s = 30.0
    cfg.attr_loss.end_s = -1.0
    cfg.attr_loss.duration_s = -1
    cfg.attr_loss.skip_steps_s = -1

    # regularizers
    cfg.reg = CN()
    cfg.reg.ow = False
    cfg.reg.ow_beta = 1e-3
    cfg.reg.nw = False
    cfg.reg.nw_max_ratio = 10.0
    cfg.reg.nw_scale = 10.0
    cfg.reg.hd = False
    cfg.reg.hd_max_score = 0.5
    cfg.reg.hd_scale = 10.0
    cfg.reg.of = False
    cfg.reg.of_beta = 1e-6
    cfg.reg.of_start_epoch = 23

    # test
    cfg.test = CN()
    cfg.test.batch_size = 100
    cfg.test.dist_metric = 'euclidean'  # distance metric, ['euclidean', 'cosine']
    cfg.test.normalize_feature = False  # normalize feature vectors before computing distance
    cfg.test.ranks = [1, 5, 10, 20]  # cmc ranks
    cfg.test.evaluate = False  # test only
    cfg.test.eval_freq = -1  # evaluation frequency (-1 means to only test after training)
    cfg.test.start_eval = 0  # start to evaluate after a specific epoch
    cfg.test.rerank = False  # use person re-ranking
    cfg.test.visrank = False  # visualize ranked results (only available when cfg.test.evaluate=True)
    cfg.test.visrank_topk = 10  # top-k ranks to visualize
    cfg.test.visactmap = False  # visualize CNN activation maps
    cfg.test.apply_masks = False

    # Augmentations
    cfg.data.transforms = CN()

    cfg.data.transforms.random_flip = CN()
    cfg.data.transforms.random_flip.enable = True
    cfg.data.transforms.random_flip.p = 0.5

    cfg.data.transforms.random_crop = CN()
    cfg.data.transforms.random_crop.enable = False
    cfg.data.transforms.random_crop.p = 0.5
    cfg.data.transforms.random_crop.scale = 0.9
    cfg.data.transforms.random_crop.margin = 0
    cfg.data.transforms.random_crop.static = False
    cfg.data.transforms.random_crop.align_ar = False
    cfg.data.transforms.random_crop.align_center = False

    cfg.data.transforms.crop_pad = CN()
    cfg.data.transforms.crop_pad.enable = False

    cfg.data.transforms.center_crop = CN()
    cfg.data.transforms.center_crop.enable = False
    cfg.data.transforms.center_crop.margin = 0
    cfg.data.transforms.center_crop.test_only = False

    cfg.data.transforms.random_gray_scale = CN()
    cfg.data.transforms.random_gray_scale.enable = False
    cfg.data.transforms.random_gray_scale.p = 0.5

    cfg.data.transforms.force_gray_scale = CN()
    cfg.data.transforms.force_gray_scale.enable = False

    cfg.data.transforms.random_negative = CN()
    cfg.data.transforms.random_negative.enable = False
    cfg.data.transforms.random_negative.p = 0.5

    cfg.data.transforms.posterize = CN()
    cfg.data.transforms.posterize.enable = False
    cfg.data.transforms.posterize.p = 0.5
    cfg.data.transforms.posterize.bits = 1

    cfg.data.transforms.equalize = CN()
    cfg.data.transforms.equalize.enable = False
    cfg.data.transforms.equalize.p = 0.5

    cfg.data.transforms.random_padding = CN()
    cfg.data.transforms.random_padding.enable = False
    cfg.data.transforms.random_padding.p = 0.5
    cfg.data.transforms.random_padding.padding = (0, 10)

    cfg.data.transforms.random_perspective = CN()
    cfg.data.transforms.random_perspective.enable = False
    cfg.data.transforms.random_perspective.p = 0.5
    cfg.data.transforms.random_perspective.distortion_scale = 0.5

    cfg.data.transforms.color_jitter = CN()
    cfg.data.transforms.color_jitter.enable = False
    cfg.data.transforms.color_jitter.p = 0.5
    cfg.data.transforms.color_jitter.brightness = 0.2
    cfg.data.transforms.color_jitter.contrast = 0.2
    cfg.data.transforms.color_jitter.saturation = 0.1
    cfg.data.transforms.color_jitter.hue = 0.1

    cfg.data.transforms.random_erase = CN()
    cfg.data.transforms.random_erase.enable = False
    cfg.data.transforms.random_erase.p = 0.5
    cfg.data.transforms.random_erase.sl = 0.2
    cfg.data.transforms.random_erase.sh = 0.4
    cfg.data.transforms.random_erase.rl = 0.3
    cfg.data.transforms.random_erase.rh = 3.3
    cfg.data.transforms.random_erase.fill_color = (125.307, 122.961, 113.8575)
    cfg.data.transforms.random_erase.norm_image = True

    cfg.data.transforms.coarse_dropout = CN()
    cfg.data.transforms.coarse_dropout.enable = False
    cfg.data.transforms.coarse_dropout.p = 0.5
    cfg.data.transforms.coarse_dropout.max_height = 8
    cfg.data.transforms.coarse_dropout.max_width = 8
    cfg.data.transforms.coarse_dropout.max_holes = 8
    cfg.data.transforms.coarse_dropout.min_holes = None
    cfg.data.transforms.coarse_dropout.min_height = None
    cfg.data.transforms.coarse_dropout.fill_value = 0
    cfg.data.transforms.coarse_dropout.mask_fill_value = None

    cfg.data.transforms.random_rotate = CN()
    cfg.data.transforms.random_rotate.enable = False
    cfg.data.transforms.random_rotate.p = 0.5
    cfg.data.transforms.random_rotate.angle = (-5, 5)
    cfg.data.transforms.random_rotate.values = (0, )

    cfg.data.transforms.cut_out_with_prior = CN()
    cfg.data.transforms.cut_out_with_prior.enable = False
    cfg.data.transforms.cut_out_with_prior.p = 0.5
    cfg.data.transforms.cut_out_with_prior.max_area = 0.1

    cfg.data.transforms.random_blur = CN()
    cfg.data.transforms.random_blur.enable = False
    cfg.data.transforms.random_blur.p = 0.5
    cfg.data.transforms.random_blur.k = 5

    cfg.data.transforms.random_noise = CN()
    cfg.data.transforms.random_noise.enable = False
    cfg.data.transforms.random_noise.p = 0.2
    cfg.data.transforms.random_noise.sigma = 0.05
    cfg.data.transforms.random_noise.grayscale = False

    cfg.data.transforms.augmix = CN()
    cfg.data.transforms.augmix.enable = False
    cfg.data.transforms.augmix.cfg_str = "augmix-m5-w3"
    cfg.data.transforms.augmix.grey_imgs = False

    cfg.data.transforms.randaugment = CN()
    cfg.data.transforms.randaugment.enable = False
    cfg.data.transforms.randaugment.p = 1.

    cfg.data.transforms.cutout = CN()
    cfg.data.transforms.cutout.enable = False
    cfg.data.transforms.cutout.p = 0.5
    cfg.data.transforms.cutout.cutout_factor=0.3
    cfg.data.transforms.cutout.fill_color='random'

    cfg.data.transforms.random_figures = CN()
    cfg.data.transforms.random_figures.enable = False
    cfg.data.transforms.random_figures.p = 0.5
    cfg.data.transforms.random_figures.random_color = True
    cfg.data.transforms.random_figures.always_single_figure = False
    cfg.data.transforms.random_figures.thicknesses = (1, 6)
    cfg.data.transforms.random_figures.circle_radiuses = (5, 64)
    cfg.data.transforms.random_figures.figure_prob = 0.5
    cfg.data.transforms.random_figures.before_resize = True
    cfg.data.transforms.random_figures.figures = ['line', 'rectangle', 'circle']

    cfg.data.transforms.random_patch = CN()
    cfg.data.transforms.random_patch.enable = False
    cfg.data.transforms.random_patch.p = 0.5
    cfg.data.transforms.random_patch.pool_capacity = 50000
    cfg.data.transforms.random_patch.min_sample_size = 100
    cfg.data.transforms.random_patch.patch_min_area = 0.01
    cfg.data.transforms.random_patch.patch_max_area = 0.5
    cfg.data.transforms.random_patch.patch_min_ratio = 0.1
    cfg.data.transforms.random_patch.prob_rotate = 0.5
    cfg.data.transforms.random_patch.prob_flip_leftright = 0.5

    cfg.data.transforms.random_grid = CN()
    cfg.data.transforms.random_grid.enable = False
    cfg.data.transforms.random_grid.p = 0.33
    cfg.data.transforms.random_grid.color = (-1, -1, -1)
    cfg.data.transforms.random_grid.grid_size = (24, 64)
    cfg.data.transforms.random_grid.thickness = (1, 1)
    cfg.data.transforms.random_grid.angle = (0, 180)
    cfg.data.transforms.random_grid.before_resize = True

    cfg.data.transforms.random_background_substitution = CN()
    cfg.data.transforms.random_background_substitution.enable = False
    cfg.data.transforms.random_background_substitution.p = 0.2
    cfg.data.transforms.random_background_substitution.images_root_dir = ''
    cfg.data.transforms.random_background_substitution.images_list_file = ''

    cfg.data.transforms.mixup = CN()
    cfg.data.transforms.mixup.enable = False
    cfg.data.transforms.mixup.p = 0.33
    cfg.data.transforms.mixup.alpha = 0.2
    cfg.data.transforms.mixup.images_root_dir = ''
    cfg.data.transforms.mixup.images_list_file = ''

    cfg.data.transforms.test = CN()
    cfg.data.transforms.test.resize_first = False
    cfg.data.transforms.test.resize_scale = 1.0

    # NNCF part
    cfg.nncf = CN()
    # this flag forces using NNCF
    cfg.nncf.enable = False
    # coefficient to decrease LR for NNCF training
    # (the original initial LR for training will be read from the checkpoint's metainfo)
    cfg.nncf.coeff_decrease_lr_for_nncf = 0.035
    # path to a json file with NNCF config
    cfg.nncf.nncf_config_path = ''
    # path to a YAML file with changes to be applied to the main train config for NNCF training
    cfg.nncf.changes_in_main_train_config = ''
    # path to a YAML file with changes to be applied to the aux train config for NNCF training
    cfg.nncf.changes_in_aux_train_config = ''

    return cfg

def merge_from_files_with_base(cfg, cfg_path):
    def _get_list_of_files(cur_path, set_of_files=None):
        if not (cur_path.lower().endswith('.yml') or cur_path.lower().endswith('.yaml')):
                raise RuntimeError(f'Wrong extension of config file {cur_path}')
        if set_of_files is None:
            set_of_files = {cur_path}
        elif cur_path in set_of_files:
            raise RuntimeError(f'Cyclic inheritance of config files found in {cur_path}')
        set_of_files.add(cur_path)

        if not os.path.isfile(cur_path):
            raise FileNotFoundError(f'Config file {cur_path} not found')

        with open(cur_path) as f:
            d = yaml.safe_load(f)

        base = d.get('_base_')
        if not base:
            return [cur_path]

        if not isinstance(base, (list, str)):
            raise RuntimeError(f'Wrong type of field "_base_" in config {cur_path}')

        if isinstance(base, list) and len(base) > 1:
            raise NotImplementedError(f'Multiple inheritance of configs is not implemented. '
                                      f'Please, fix the config {cur_path}')
        if isinstance(base, list):
            base = base[0]
            if not isinstance(base, str):
                raise RuntimeError(f'Wrong type of the element in the field "_base_" in config {cur_path}')


        cur_list_files = _get_list_of_files(base, set_of_files)
        cur_list_files += [cur_path]
        return cur_list_files

    cur_list_files = _get_list_of_files(cfg_path)
    assert len(cur_list_files) >= 1

    print('Begin merging of config files with inheritance')
    for cur_path in cur_list_files:
        print(f'    merging config file {cur_path}')
        cfg.merge_from_file(cur_path)
    print('End merging of config files with inheritance')


def imagedata_kwargs(cfg):
    return {
        'root': cfg.data.root,
        'sources': cfg.data.sources,
        'targets': cfg.data.targets,
        'height': cfg.data.height,
        'width': cfg.data.width,
        'transforms': cfg.data.transforms,
        'norm_mean': cfg.data.norm_mean,
        'norm_std': cfg.data.norm_std,
        'use_gpu': cfg.use_gpu,
        'split_id': cfg.data.split_id,
        'combineall': cfg.data.combineall,
        'batch_size_train': cfg.train.batch_size,
        'batch_size_test': cfg.test.batch_size,
        'workers': cfg.data.workers,
        'batch_num_instances': cfg.sampler.batch_num_instances,
        'epoch_num_instances': cfg.sampler.epoch_num_instances,
        'fill_instances': cfg.sampler.fill_instances,
        'train_sampler': cfg.sampler.train_sampler,
        'enable_masks': cfg.data.enable_masks,
        'num_sampled_packages': cfg.data.num_sampled_packages,
        # image
        'cuhk03_labeled': cfg.cuhk03.labeled_images,
        'cuhk03_classic_split': cfg.cuhk03.classic_split,
        'market1501_500k': cfg.market1501.use_500k_distractors,
        'apply_masks_to_test': cfg.test.apply_masks,
        'min_samples_per_id': cfg.data.min_samples_per_id,
        'custom_dataset_names': cfg.custom_datasets.names,
        'custom_dataset_roots': cfg.custom_datasets.roots,
        'custom_dataset_types': cfg.custom_datasets.types,
    }


def optimizer_kwargs(cfg):
    return {
        'optim': cfg.train.optim,
        'base_optim': cfg.train.base_optim,
        'lr': cfg.train.lr,
        'weight_decay': cfg.train.weight_decay,
        'momentum': cfg.sgd.momentum,
        'sgd_dampening': cfg.sgd.dampening,
        'sgd_nesterov': cfg.sgd.nesterov,
        'rmsprop_alpha': cfg.rmsprop.alpha,
        'adam_beta1': cfg.adam.beta1,
        'adam_beta2': cfg.adam.beta2,
        'staged_lr': cfg.train.staged_lr,
        'new_layers': cfg.train.new_layers,
        'base_lr_mult': cfg.train.base_lr_mult,
        'nbd': cfg.train.nbd,
        'lr_finder': cfg.lr_finder.enable,
        'sam_rho': cfg.train.sam.rho
    }


def lr_scheduler_kwargs(cfg):
    return {
        'lr_scheduler': cfg.train.lr_scheduler,
        'base_scheduler': cfg.train.base_scheduler,
        'stepsize': cfg.train.stepsize,
        'gamma': cfg.train.gamma,
        'max_epoch': cfg.train.max_epoch,
        'warmup': cfg.train.warmup,
        'multiplier': cfg.train.multiplier,
        'first_cycle_steps': cfg.train.first_cycle_steps,
        'cycle_mult': cfg.train.cycle_mult,
        'min_lr': cfg.train.min_lr,
        'max_lr': cfg.train.max_lr,
        'patience': cfg.train.patience,
        'pct_start' : cfg.train.pct_start,
        'lr_decay_factor': cfg.train.lr_decay_factor,
    }


def model_kwargs(cfg, num_classes):
    if isinstance(num_classes, (tuple, list)) and len(num_classes) == 1:
        num_classes = num_classes[0]

    return {
        'name': cfg.model.name,
        'num_classes': num_classes,
        'loss': cfg.loss.name,
        'pretrained': cfg.model.pretrained,
        'lr_finder': cfg.lr_finder,
        'download_weights': cfg.model.download_weights,
        'use_gpu': cfg.use_gpu,
        'dropout_cfg': cfg.model.dropout_backbone,
        'dropout_cls': cfg.model.dropout_cls,
        'feature_dim': cfg.model.feature_dim,
        'fpn_cfg': cfg.model.fpn,
        'pooling_type': cfg.model.pooling_type,
        'input_size': (cfg.data.height, cfg.data.width),
        'IN_first': cfg.model.IN_first,
        'IN_conv1': cfg.model.IN_conv1,
        'extra_blocks': cfg.model.extra_blocks,
        'lct_gate': cfg.model.lct_gate,
        'bn_eval': cfg.model.bn_eval,
        'bn_frozen': cfg.model.bn_frozen,
        'enable_attentions': cfg.model.enable_attentions and cfg.data.enable_masks,
        'attr_names': cfg.attr_loss.names,
        'attr_num_classes': cfg.attr_loss.num_classes,
        'type': cfg.model.type,
        'self_challenging_cfg': cfg.model.self_challenging_cfg,
    }


def engine_run_kwargs(cfg):
    return {
        'save_dir': cfg.data.save_dir,
        'tb_log_dir': cfg.data.tb_log_dir,
        'max_epoch': cfg.train.max_epoch,
        'start_epoch': cfg.train.start_epoch,
        'fixbase_epoch': cfg.train.fixbase_epoch,
        'open_layers': cfg.train.open_layers,
        'start_eval': cfg.test.start_eval,
        'eval_freq': cfg.test.eval_freq,
        'test_only': cfg.test.evaluate,
        'print_freq': cfg.train.print_freq,
        'dist_metric': cfg.test.dist_metric,
        'normalize_feature': cfg.test.normalize_feature,
        'visrank': cfg.test.visrank,
        'visrank_topk': cfg.test.visrank_topk,
        'use_metric_cuhk03': cfg.cuhk03.use_metric_cuhk03,
        'ranks': cfg.test.ranks,
        'rerank': cfg.test.rerank,
        'initial_seed': cfg.train.seed
    }


def lr_finder_run_kwargs(cfg):
    return {
        'mode': cfg.lr_finder.mode,
        'epochs_warmup': cfg.lr_finder.epochs_warmup,
        'max_lr': cfg.lr_finder.max_lr,
        'min_lr': cfg.lr_finder.min_lr,
        'step': cfg.lr_finder.step,
        'num_epochs': cfg.lr_finder.num_epochs,
        'path_to_savefig': cfg.lr_finder.path_to_savefig,
        'seed': cfg.train.seed,
        'smooth_f': cfg.lr_finder.smooth_f,
        'n_trials': cfg.lr_finder.n_trials
    }

def transforms(cfg):
    return cfg.data.transforms


def augmentation_kwargs(cfg):
    return {
        'random_flip': cfg.data.transforms.random_flip,
        'center_crop': cfg.data.transforms.center_crop,
        'random_crop': cfg.data.transforms.random_crop,
        'random_gray_scale': cfg.data.transforms.random_gray_scale,
        'force_gray_scale': cfg.data.transforms.force_gray_scale,
        'random_padding': cfg.data.transforms.random_padding,
        'random_perspective': cfg.data.transforms.random_perspective,
        'color_jitter': cfg.data.transforms.color_jitter,
        'random_erase': cfg.data.transforms.random_erase,
        'random_rotate': cfg.data.transforms.random_rotate,
        'random_figures': cfg.data.transforms.random_figures,
        'random_grid': cfg.data.transforms.random_grid,
        'random_negative': cfg.data.transforms.random_negative,
        'cut_out_with_prior': cfg.data.transforms.cut_out_with_prior,
        'coarse_dropout': cfg.data.transforms.coarse_dropout,
        'equalize': cfg.data.transforms.equalize,
        'posterize': cfg.data.transforms.posterize,
        'augmix': cfg.data.transforms.augmix
    }
