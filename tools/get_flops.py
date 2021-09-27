import argparse
import json

import torch
from scripts.default_config import (get_default_config, imagedata_kwargs,
                                    model_kwargs, merge_from_files_with_base)

import torchreid
from torchreid.utils import collect_env_info, set_random_seed
from ptflops import get_model_complexity_info


def build_datamanager(cfg, classification_classes_filter=None):
    return torchreid.data.ImageDataManager(filter_classes=classification_classes_filter, **imagedata_kwargs(cfg))


def reset_config(cfg, args):
    if args.root:
        cfg.data.root = args.root
    if args.custom_roots:
        cfg.custom_datasets.roots = args.custom_roots
    if args.custom_types:
        cfg.custom_datasets.types = args.custom_types
    if args.custom_names:
        cfg.custom_datasets.names = args.custom_names


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--config-file', type=str, default='', required=True,
                        help='path to config file')
    parser.add_argument('--custom-roots', type=str, nargs='+',
                        help='types or paths to annotation of custom datasets (delimited by space)')
    parser.add_argument('--custom-types', type=str, nargs='+',
                        help='path of custom datasets (delimited by space)')
    parser.add_argument('--custom-names', type=str, nargs='+',
                        help='names of custom datasets (delimited by space)')
    parser.add_argument('--root', type=str, default='', help='path to data root')
    parser.add_argument('--classes', type=str, nargs='+',
                        help='name of classes in classification dataset')
    parser.add_argument('--out')
    parser.add_argument('opts', default=None, nargs=argparse.REMAINDER,
                        help='Modify config options using the command-line')
    args = parser.parse_args()

    cfg = get_default_config()
    cfg.use_gpu = torch.cuda.is_available()
    if args.config_file:
        merge_from_files_with_base(cfg, args.config_file)
    reset_config(cfg, args)
    cfg.merge_from_list(args.opts)
    set_random_seed(cfg.train.seed)

    print('Show configuration\n{}\n'.format(cfg))
    print('Collecting env info ...')
    print('** System info **\n{}\n'.format(collect_env_info()))

    if cfg.use_gpu:
        torch.backends.cudnn.benchmark = True

    datamanager = build_datamanager(cfg, args.classes)
    num_train_classes = datamanager.num_train_pids

    print('Building main model: {}'.format(cfg.model.name))
    model = torchreid.models.build_model(**model_kwargs(cfg, num_train_classes))
    macs, num_params = get_model_complexity_info(model, (3, cfg.data.height, cfg.data.width),
                                                 as_strings=False, verbose=False, print_per_layer_stat=False)
    print('Main model complexity: M params={:,} G flops={:,}'.format(num_params / 10**6, macs * 2 / 10**9))

    if args.out:
        out = list()
        out.append({'key': 'size', 'display_name': 'Size', 'value': num_params / 10**6, 'unit': 'Mp'})
        out.append({'key': 'complexity', 'display_name': 'Complexity', 'value': 2 * macs / 10**9,
                    'unit': 'GFLOPs'})
        print('dump to' + args.out)
        with open(args.out, 'w') as write_file:
            json.dump(out, write_file, indent=4)

if __name__ == '__main__':
    main()
