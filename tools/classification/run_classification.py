import argparse
import os
import re
import tempfile
from pathlib import Path
from subprocess import run
import json

import numpy as np
from ruamel.yaml import YAML


def get_lr_sets(model_name: str):
    if "mobilenet" in model_name:
        return {"CIFAR100": 0.005, "pets": 0.005, "caltech101": 0.015,
                "cars": 0.025, "flowers": 0.02, "DTD": 0.008, "FOOD101": 0.015,
                "birdsnap": 0.015,"FashionMNIST": 0.012, "SUN397": 0.008, "SVHN": 0.015,
                "attd_mi02_v3": 0.005, "attd_mi04_v4": 0.012, "lgchem": 0.015, "autism": 0.015}

    elif 'efficientnet_b0' == model_name:
        return {"CIFAR100": 0.005, "pets": 0.005, "caltech101": 0.015,
                "cars": 0.025, "flowers": 0.02, "DTD": 0.008, "FOOD101": 0.015,
                "birdsnap": 0.015,"FashionMNIST": 0.012, "SUN397": 0.008, "SVHN": 0.015,
                "attd_mi02_v3": 0.005, "attd_mi04_v4": 0.012, "lgchem": 0.015, "autism": 0.015}
    elif 'efficientnetv2_s' in model_name:
        return {"CIFAR100": 0.004488, "pets": 0.00597, "caltech101": 0.01355,
                "cars": 0.0155, "flowers": 0.010624, "DTD": 0.009936, "FOOD101": 0.0043962,
                "birdsnap": 0.004488,"FashionMNIST": 0.0048896, "SUN397": 0.008, "SVHN": 0.015,
                "attd_mi02_v3": 0.005, "attd_mi04_v4": 0.012, "lgchem": 0.015, "autism": 0.015}
    else:
        print("Unknown model. Use stadart predefined lrs")
        return {"CIFAR100": 0.003, "pets": 0.003, "caltech101": 0.003,
                "cars": 0.003, "flowers": 0.003, "DTD": 0.003, "FOOD101": 0.003,
                "birdsnap": 0.003,"FashionMNIST": 0.003, "SUN397": 0.003, "SVHN": 0.003,
                "attd_mi02_v3": 0.003, "attd_mi04_v4": 0.003, "lgchem": 0.003, "autism": 0.003}

def read_config(yaml: YAML, config_path: str):
    yaml.default_flow_style = True
    with open(config_path, 'r') as f:
        cfg = yaml.load(f)
    return cfg

def dump_config(yaml: YAML, config_path: str, cfg: dict):
    with open(config_path, 'w') as f:
        yaml.default_flow_style = True
        yaml.dump(cfg, f)

def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument( '--root', type=str, required=False, default='/datasets/classification', help='path to folder with datasets')
    parser.add_argument('--config', type=str, required=False, help='path to config file')
    parser.add_argument('--path-to-main', type=str, default='./tools/main.py',required=False, help='path to main.py file')
    parser.add_argument('--gpu-num', type=int, default=1, help='Number of GPUs for training. 0 is for CPU mode')
    parser.add_argument('--use-hardcoded-lr', action='store_true')
    parser.add_argument('-d','--domains', nargs='+', help='On what domains to train', required=False, default=['all'])
    parser.add_argument('--dump-results', type=bool, default=True, help='whether or not to dump results of the experiment')
    args = parser.parse_args()
    yaml = YAML()

    # datasets to experiment with
    datasets = dict(
        CIFAR100=dict(
            roots=['CIFAR100/train', 'CIFAR100/val'],
            names=['CIFAR100_train', 'CIFAR100_val'],
            types=['classification_image_folder', 'classification_image_folder'],
            sources='CIFAR100_train',
            targets='CIFAR100_val',
        ),
        flowers=dict(
            roots=['flowers/train.txt', 'flowers/val.txt'],
            names=['flowers_train', 'flowers_val'],
            types=['classification', 'classification'],
            sources='flowers_train',
            targets='flowers_val',
        ),
        cars=dict(
            roots=['cars/train.txt', 'cars/val.txt'],
            names=['cars_train', 'cars_val'],
            types=['classification', 'classification'],
            sources='cars_train',
            targets='cars_val',
        ),
        DTD=dict(
            roots=['DTD/train', 'DTD/val'],
            names=['DTD_train', 'DTD_val'],
            types=['classification_image_folder', 'classification_image_folder'],
            sources='DTD_train',
            targets='DTD_val',
        ),
        pets=dict(
            roots=['pets/train.txt', 'pets/val.txt'],
            names=['pets_train', 'pets_val'],
            types=['classification', 'classification'],
            sources='pets_train',
            targets='pets_val',
        ),
        birdsnap=dict(
            roots=['birdsnap/train.txt', 'birdsnap/val.txt'],
            names=['birdsnap_train', 'birdsnap_val'],
            types=['classification', 'classification'],
            sources='birdsnap_train',
            targets='birdsnap_val',
        ),
        caltech101=dict(
            roots=['caltech101/train.txt', 'caltech101/val.txt'],
            names=['caltech101_train', 'caltech101_val'],
            types=['classification', 'classification'],
            sources='caltech101_train',
            targets='caltech101_val',
        ),
        FOOD101=dict(
            roots=['FOOD101/train.txt', 'FOOD101/val.txt'],
            names=['FOOD101_train', 'FOOD101_val'],
            types=['classification', 'classification'],
            sources='FOOD101_train',
            targets='FOOD101_val',
        ),
        lg_chem=dict(
            roots=['lg_chem/train.txt', 'lg_chem/val.txt'],
            names=['lg_chem_train', 'lg_chem_val'],
            types=['classification', 'classification'],
            sources='lg_chem_train',
            targets='lg_chem_val',
        ),
        autism=dict(
            roots=['autism/train', 'autism/val'],
            names=['autism_train', 'autism_val'],
            types=['classification_image_folder', 'classification_image_folder'],
            sources='autism_train',
            targets='autism_val',
        ),
        attd_mi04_v4=dict(
            roots=['attd_mi04_v4/train.txt', 'attd_mi04_v4/val.txt'],
            names=['attd_mi04_v4_train', 'attd_mi04_v4_val'],
            types=['classification', 'classification'],
            sources='attd_mi04_v4_train',
            targets='attd_mi04_v4_val',
        ),
        attd_mi02_v3=dict(
            roots=['attd_mi02_v3/train.txt', 'attd_mi02_v3/val.txt'],
            names=['attd_mi02_v3_train', 'attd_mi02_v3_val'],
            types=['classification', 'classification'],
            sources='attd_mi02_v3_train',
            targets='attd_mi02_v3_val',
        )
    )

    path_to_base_cfg = args.config
    # write datasets you want to skip
    domains = args.domains
    if 'all' in domains:
        domains = set(datasets.keys())

    for key in domains:
        params = datasets[key]
        cfg = read_config(yaml, path_to_base_cfg)
        lrs_dict = get_lr_sets(cfg["model"]["name"])
        if key in ["attd_mi02_v3", "attd_mi04_v4", "lg_chem", "fashionMNIST", "SVHN"]:
            cfg['data']['transforms']['augmix']['grey_imgs'] = True
        path_to_exp_folder = cfg['data']['save_dir']
        name_train = params['names'][0]
        name_val = params['names'][1]
        type_train = params['types'][0]
        type_val = params['types'][1]
        root_train = args.root + os.sep + params['roots'][0]
        root_val = args.root + os.sep + params['roots'][1]
        if args.use_hardcoded_lr:
            print("WARNING: Using hardcoded LR")
            if key in lrs_dict:
                cfg['lr_finder']["enable"] = False
                cfg["train"]["lr"] = lrs_dict[key]
            else:
                cfg['lr_finder']["enable"] = True

        cfg['custom_datasets']['roots'] = [root_train, root_val]
        cfg['custom_datasets']['types'] = [type_train, type_val]
        cfg['custom_datasets']['names'] = [name_train, name_val]

        cfg['data']['save_dir'] = path_to_exp_folder + f"/{key}"

        source = params['sources']
        targets = params['targets']
        cfg['data']['sources'] = [source]
        cfg['data']['targets'] = [targets]
        # dump it
        fd, tmp_path_to_cfg = tempfile.mkstemp(suffix='.yml')
        try:
            with os.fdopen(fd, 'w') as tmp:
                # do stuff with temp file
                yaml.default_flow_style = True
                yaml.dump(cfg, tmp)
                tmp.close()

            # run training
            run(['python', f'{str(args.path_to_main)}',
                '--config', f'{tmp_path_to_cfg}',
                '--gpu-num', f'{int(args.gpu_num)}'],
                shell=False)
        finally:
            os.remove(tmp_path_to_cfg)
    # after training combine all outputs in one file
    if args.dump_results:
        path_to_bash = str(Path.cwd() / 'tools/classification/parse_output.sh')
        run(['bash', f'{path_to_bash}', f'{path_to_exp_folder}'], shell=False)
        saver = dict()
        path_to_file = f"{path_to_exp_folder}/combine_all.txt"
        # parse output file from bash script
        with open(path_to_file, 'r') as f:
            for line in f:
                if line.strip() in datasets.keys():
                    next_dataset = line.strip()
                    saver[next_dataset] = dict()
                    continue
                else:
                    for metric in ['mAP', 'Rank-1', 'Rank-5']:
                        if line.strip().startswith(metric):
                            if not metric in saver[next_dataset]:
                                saver[next_dataset][metric] = []
                            pattern = re.search('\d+\.\d+', line.strip())
                            if pattern:
                                saver[next_dataset][metric].append(
                                    float(pattern.group(0))
                                )

        # dump in appropriate patern
        names = ''
        values = ''
        with open(path_to_file, 'a') as f:
            for key in sorted(datasets.keys()):
                names += key + ' '
                if key in saver:
                    best_top_1_idx = np.argmax(saver[key]['Rank-1'])
                    top1 = str(saver[key]['Rank-1'][best_top_1_idx])
                    mAP = str(saver[key]['mAP'][best_top_1_idx])
                    top5 = str(saver[key]['Rank-5'][best_top_1_idx])
                    values += mAP + ';' + top1 + ';' + top5 + ';'
                else:
                    values += '-1;-1;-1;'

            f.write(f"\n{names}\n{values}")

if __name__ == "__main__":
    main()
