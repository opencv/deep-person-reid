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
        return {"COCO": 0.0001, "VOC": 0.0002,"NUS": 0.0001, "VG500": 0.0001}
    elif 'efficientnetv2' in model_name:
        return {"COCO": 0.0001, "VOC": 0.0001,"NUS": 0.0001, "VG500": 0.0001}
    else:
        print("Unknown model. Use stadart predefined lrs")
        return {"COCO": 0.0001, "VOC": 0.0001,"NUS": 0.0001, "VG500": 0.0001}

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
    parser.add_argument( '--root', type=str, required=False, default='/ssd/datasets', help='path to folder with datasets')
    parser.add_argument('--config', type=str, required=False, help='path to config file')
    parser.add_argument('--path-to-main', type=str, default='./tools/main.py',required=False, help='path to main.py file')
    parser.add_argument('--gpu-num', type=int, default=1, help='Number of GPUs for training. 0 is for CPU mode')
    parser.add_argument('--use-hardcoded-lr', action='store_true')
    parser.add_argument('-d','--domains', nargs='+', help='On what domains to train', required=False, default=['all'])
    parser.add_argument('-lrs', '--lr-sets', type=json.loads, default='{"COCO": 0.0001, "VOC": 0.0002,'
                                                                            '"NUS": 0.0001, "VG500": 0.0001}')
    parser.add_argument('--dump-results', type=bool, default=True, help='whether or not to dump results of the experiment')
    args = parser.parse_args()
    yaml = YAML()

    # datasets to experiment with
    datasets = dict(
        coco=dict(
            roots=['coco/train.json', 'coco/val.json'],
            names=['coco_train', 'coco_val'],
            types=['multilabel_classification', 'multilabel_classification'],
            sources='coco_train',
            targets='coco_val',
        ),
        voc=dict(
            roots=['mlc_voc_2007/train.json', 'mlc_voc_2007/val.json'],
            names=['voc_train', 'voc_val'],
            types=['multilabel_classification', 'multilabel_classification'],
            sources='voc_train',
            targets='voc_val',
        ),
        vg500=dict(
            roots=['VG500/train.json', 'VG500/val.json'],
            names=['VG500_train', 'VG500_val'],
            types=['multilabel_classification', 'multilabel_classification'],
            sources='VG500_train',
            targets='VG500_val',
        ),
        nus_wide=dict(
            roots=['nus_wide/train.json', 'nus_wide/val.json'],
            names=['nus_wide_train', 'nus_wide_val'],
            types=['multilabel_classification', 'multilabel_classification'],
            sources='nus_wide_train',
            targets='nus_wide_val',
        ),
        pets=dict(
            roots=['oxford_pets/train.json', 'oxford_pets/val.json'],
            names=['oxford_pets_train', 'oxford_pets_val'],
            types=['multilabel_classification', 'multilabel_classification'],
            sources='oxford_pets_train',
            targets='oxford_pets_val',
        ),
        bbcd=dict(
            roots=['BBCD/train.json', 'BBCD/val.json'],
            names=['bbcd_train', 'bbcd_val'],
            types=['multilabel_classification', 'multilabel_classification'],
            sources='bbcd_train',
            targets='bbcd_val',
        ),
        aerial_maritime=dict(
            roots=['Aerial_Maritime/train.json', 'Aerial_Maritime/val.json'],
            names=['aerial_maritime_train', 'aerial_maritime_val'],
            types=['multilabel_classification', 'multilabel_classification'],
            sources='aerial_maritime_train',
            targets='aerial_maritime_val',
        ),
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
                    for metric in ['mAP', 'F_O', 'mean_F_C']:
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
                    best_top_1_idx = np.argmax(saver[key]['F_O'])
                    fo = str(saver[key]['F_O'][best_top_1_idx])
                    mAP = str(saver[key]['mAP'][best_top_1_idx])
                    fc = str(saver[key]['mean_F_C'][best_top_1_idx])
                    values += mAP + ';' + fo + ';' + fc + ';'
                else:
                    values += '-1;-1;-1;'

            f.write(f"\n{names}\n{values}")

if __name__ == "__main__":
    main()
