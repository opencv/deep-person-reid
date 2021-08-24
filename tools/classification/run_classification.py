import argparse
import os
import re
import tempfile
from pathlib import Path
from subprocess import run

import numpy as np
from ruamel.yaml import YAML


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
    parser.add_argument('--dump-results', type=bool, default=True, help='whether or not to dump results of the experiment')
    args = parser.parse_args()
    yaml = YAML()

    # datasets to experiment with
    datasets = dict(
        CIFAR100=dict(
            resolution=(224, 224),
            epochs=35,
            roots=['CIFAR100/train', 'CIFAR100/val'],
            names=['CIFAR100_train', 'CIFAR100_val'],
            types=['classification_image_folder', 'classification_image_folder'],
            sources='CIFAR100_train',
            targets='CIFAR100_val',
            batch_size=128,
            num_C=100
        ),
        SUN397=dict(
            resolution=(224, 224),
            epochs=60,
            roots=['SUN397/train.txt', 'SUN397/val.txt'],
            names=['SUN397_train', 'SUN397_val'],
            types=['classification', 'classification'],
            sources='SUN397_train',
            targets='SUN397_val',
            batch_size=128,
            num_C=397
        ),
        flowers=dict(
            resolution=(224, 224),
            epochs=50,
            roots=['flowers/train.txt', 'flowers/val.txt'],
            names=['flowers_train', 'flowers_val'],
            types=['classification', 'classification'],
            sources='flowers_train',
            targets='flowers_val',
            batch_size=128,
            num_C=102
        ),
        fashionMNIST=dict(
            resolution=(224, 224),
            epochs=35,
            roots=['fashionMNIST/train', 'fashionMNIST/val'],
            names=['fashionMNIST_train', 'fashionMNIST_val'],
            types=['classification_image_folder', 'classification_image_folder'],
            sources='fashionMNIST_train',
            targets='fashionMNIST_val',
            batch_size=128,
            num_C=10
        ),
        SVHN=dict(
            resolution=(224, 224),
            epochs=50,
            roots=['SVHN/train', 'SVHN/val'],
            names=['SVHN_train', 'SVHN_val'],
            types=['classification_image_folder', 'classification_image_folder'],
            sources='SVHN_train',
            targets='SVHN_val',
            batch_size=128,
            num_C=10
        ),
        cars=dict(
            resolution=(224, 224),
            epochs=110,
            roots=['cars/train.txt', 'cars/val.txt'],
            names=['cars_train', 'cars_val'],
            types=['classification', 'classification'],
            sources='cars_train',
            targets='cars_val',
            batch_size=128,
            num_C=196
        ),
        DTD=dict(
            resolution=(224, 224),
            epochs=70,
            roots=['DTD/train', 'DTD/val'],
            names=['DTD_train', 'DTD_val'],
            types=['classification_image_folder', 'classification_image_folder'],
            sources='DTD_train',
            targets='DTD_val',
            batch_size=128,
            num_C=47
        ),
        pets=dict(
            resolution=(224, 224),
            epochs=60,
            roots=['pets/train.txt', 'pets/val.txt'],
            names=['pets_train', 'pets_val'],
            types=['classification', 'classification'],
            sources='pets_train',
            targets='pets_val',
            batch_size=128,
            num_C=37
        ),
        Xray=dict(
            resolution=(224, 224),
            epochs=70,
            roots=['Xray/train', 'Xray/val'],
            names=['Xray_train', 'Xray_val'],
            types=['classification_image_folder', 'classification_image_folder'],
            sources='Xray_train',
            targets='Xray_val',
            batch_size=128,
            num_C=2
        ),
        birdsnap=dict(
            resolution=(224, 224),
            epochs=35,
            roots=['birdsnap/train.txt', 'birdsnap/val.txt'],
            names=['birdsnap_train', 'birdsnap_val'],
            types=['classification', 'classification'],
            sources='birdsnap_train',
            targets='birdsnap_val',
            batch_size=128,
            num_C=500
        ),
        caltech101=dict(
            resolution=(224, 224),
            epochs=55,
            roots=['caltech101/train.txt', 'caltech101/val.txt'],
            names=['caltech101_train', 'caltech101_val'],
            types=['classification', 'classification'],
            sources='caltech101_train',
            targets='caltech101_val',
            batch_size=128,
            num_C=101
        ),
        FOOD101=dict(
            resolution=(224, 224),
            epochs=35,
            roots=['FOOD101/train.txt', 'FOOD101/val.txt'],
            names=['FOOD101_train', 'FOOD101_val'],
            types=['classification', 'classification'],
            sources='FOOD101_train',
            targets='FOOD101_val',
            batch_size=128,
            num_C=101
        ),
        LGChenck=dict(
            resolution=(224, 224),
            roots=['LGChenck/Dataset_1', 'LGChenck/Dataset_2'],
            names=['LGChenck_train', 'LGChenck_val'],
            types=['classification_image_folder', 'classification_image_folder'],
            sources='LGChenck_train',
            targets='LGChenck_val',
            batch_size=128,
            num_C=101
        ),
        brain_tumor=dict(
            resolution=(224, 224),
            epochs=35,
            roots=['brain_tumor/train.txt', 'brain_tumor/val.txt'],
            names=['brain_tumor_train', 'brain_tumor_val'],
            types=['classification', 'classification'],
            sources='brain_tumor_train',
            targets='brain_tumor_val',
            batch_size=128,
            num_C=101
        ),
        autism=dict(
            resolution=(224, 224),
            epochs=35,
            roots=['autism/train', 'autism/val'],
            names=['autism_train', 'autism_val'],
            types=['classification_image_folder', 'classification_image_folder'],
            sources='autism_train',
            targets='autism_val',
            batch_size=128,
            num_C=101
        ),
        medicalMNIST=dict(
            resolution=(224, 224),
            epochs=35,
            roots=['medicalMNIST/train.txt', 'medicalMNIST/val.txt'],
            names=['medicalMNIST_train', 'medicalMNIST_val'],
            types=['classification', 'classification'],
            sources='medicalMNIST_train',
            targets='medicalMNIST_val',
            batch_size=128,
            num_C=101
        ),
        Covid19=dict(
            resolution=(224, 224),
            epochs=35,
            roots=['Covid19/train', 'Covid19/val'],
            names=['Covid19_train', 'Covid19_val'],
            types=['classification_image_folder', 'classification_image_folder'],
            sources='Covid19_train',
            targets='Covid19_val',
            batch_size=128,
            num_C=101
        ),
        attd_mi04_v4=dict(
            resolution=(224, 224),
            epochs=35,
            roots=['attd_mi04_v4/train.txt', 'attd_mi04_v4/val.txt'],
            names=['attd_mi04_v4_train', 'attd_mi04_v4_val'],
            types=['classification', 'classification'],
            sources='attd_mi04_v4_train',
            targets='attd_mi04_v4_val',
            batch_size=128,
            num_C=101
        ),
        attd_mi02_v3=dict(
            resolution=(224, 224),
            epochs=35,
            roots=['attd_mi02_v3/train.txt', 'attd_mi02_v3/val.txt'],
            names=['attd_mi02_v3_train', 'attd_mi02_v3_val'],
            types=['classification', 'classification'],
            sources='attd_mi02_v3_train',
            targets='attd_mi02_v3_val',
            batch_size=128,
            num_C=101
        )
    )

    path_to_base_cfg = args.config
    # write datasets you want to skip
    to_train = {'pets', 'caltech101', 'cars', 'CIFAR100', 'DTD', 'FOOD101'}

    for key, params in datasets.items():
        if key not in to_train:
            continue
        cfg = read_config(yaml, path_to_base_cfg)
        path_to_exp_folder = cfg['data']['save_dir']
        name_train = params['names'][0]
        name_val = params['names'][1]
        type_train = params['types'][0]
        type_val = params['types'][1]
        root_train = args.root + os.sep + params['roots'][0]
        root_val = args.root + os.sep + params['roots'][1]
        if args.use_hardcoded_lr:
            cfg['lr_finder']["enable"] = False
            print("WARNING: Using hardcoded LR")
            if key in ["cars", "caltech101"]:
                cfg["train"]["lr"] = 0.025
            elif key in ["pets", "CIFAR100"]:
                cfg["train"]["lr"] = 0.005
            elif key in ["flowers"]:
                cfg['train']['lr'] = 0.02
            elif key in ["DTD"]:
                cfg["train"]["lr"] = 0.008
            elif key in ["LGChenck", "autism", 'FOOD101']:
                cfg["train"]["lr"] = 0.015
            elif key in ["attd_mi02_v3"]:
                cfg["train"]["lr"] = 0.005
            elif key in ["attd_mi04_v4"]:
                cfg["train"]["lr"] = 0.012


        cfg['custom_datasets']['roots'] = [root_train, root_val]
        cfg['custom_datasets']['types'] = [type_train, type_val]
        cfg['custom_datasets']['names'] = [name_train, name_val]

        cfg['data']['height'] = params['resolution'][0]
        cfg['data']['width'] = params['resolution'][1]
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
            run(
                f'python {str(args.path_to_main)}'
                f' --config {tmp_path_to_cfg}'
                f' --gpu-num {int(args.gpu_num)}',
                shell=True
                )
        finally:
            os.remove(tmp_path_to_cfg)
    # after training combine all outputs in one file
    if args.dump_results:
        path_to_bash = str(Path.cwd() / 'parse_output.sh')
        run(f'bash {path_to_bash} {path_to_exp_folder}', shell=True)
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
                    snapshot = str(best_top_1_idx)
                    values += mAP + ';' + top1 + ';' + top5 + ';' + snapshot + ';'
                else:
                    values += '-1;-1;-1;-1;'

            f.write(f"\n{names}\n{values}")

if __name__ == "__main__":
    main()
