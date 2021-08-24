from __future__ import absolute_import, division, print_function
import os
import os.path as osp
import json

from PIL import Image

from ..dataset import ImageDataset


class Classification(ImageDataset):
    """Classification dataset.
    """

    def __init__(self, root='', mode='train', dataset_id=0, load_masks=False, **kwargs):
        if load_masks:
            raise NotImplementedError

        self.root = osp.abspath(osp.expanduser(root))
        self.data_dir = osp.dirname(self.root)
        self.annot = self.root

        required_files = [
            self.data_dir, self.annot
        ]
        self.check_before_run(required_files)

        if mode == 'train':
            train = self.load_annotation(
                self.annot,
                self.data_dir,
                dataset_id=dataset_id
            )
        else:
            train = []

        if mode == 'query':
            query = self.load_annotation(
                self.annot,
                self.data_dir,
                dataset_id=dataset_id
            )
        else:
            query = []

        gallery = []

        super(Classification, self).__init__(train, query, gallery, mode=mode, **kwargs)

    @staticmethod
    def load_annotation(annot_path, data_dir, dataset_id=0):
        out_data = []
        for line in open(annot_path):
            parts = line.strip().split(' ')
            if len(parts) != 2:
                print("line doesn't fits pattern. Expected: 'relative_path/to/image label'")
                continue
            rel_image_path, label_str = parts
            full_image_path = osp.join(data_dir, rel_image_path)
            if not osp.exists(full_image_path):
                print(f"{full_image_path}: doesn't exist. Please check path or file")
                continue

            label = int(label_str)
            out_data.append((full_image_path, label, 0, dataset_id, '', -1, -1))
        return out_data


class ExternalDatasetWrapper(ImageDataset):
    def __init__(self, data_provider, mode='train', dataset_id=0, load_masks=False, filter_classes=None, **kwargs):
        if load_masks:
            raise NotImplementedError
        self.data_provider = data_provider

        if mode == 'train':
            train, classes = self.load_annotation(
                self.data_provider, filter_classes, dataset_id
            )
            query = []
        elif mode == 'query':
            query, classes = self.load_annotation(
                self.data_provider, filter_classes, dataset_id
            )
            train = []
        else:
            classes = []
            train, query = [], []

        gallery = []

        super().__init__(train, query, gallery, mode=mode, **kwargs)

        self.classes = classes

    def __len__(self):
        return len(self.data_provider)

    def get_input(self, idx: int):
        sample = self.data_provider[idx]['img']

        if self.transform is not None:
            img = Image.fromarray(sample)
            img, _ = self.transform((img, ''))
        return img

    def __getitem__(self, idx: int):
        input_image = self.get_input(idx)
        label = self.data_provider[idx]['label']
        return input_image, label, 0

    @staticmethod
    def load_annotation(data_provider, filter_classes=None, dataset_id=0):

        all_classes = sorted(data_provider.get_classes())
        class_to_idx = {all_classes[i]: i for i in range(len(all_classes))}

        all_annotation = data_provider.get_annotation()
        out_data = []
        for item in all_annotation:
            out_data.append(('', item['label'], 0, dataset_id, '', -1, -1))\

        return out_data, class_to_idx


class ClassificationImageFolder(ImageDataset):
    """Classification dataset representing raw folders without annotation files.
    """

    def __init__(self, root='', mode='train', dataset_id=0, load_masks=False, filter_classes=None, **kwargs):
        if load_masks:
            raise NotImplementedError

        self.root = osp.abspath(osp.expanduser(root))

        required_files = [
            self.root
        ]
        self.check_before_run(required_files)

        if mode == 'train':
            train, classes = self.load_annotation(
                self.root, filter_classes, dataset_id
            )
            query = []
        elif mode == 'query':
            query, classes = self.load_annotation(
                self.root, filter_classes, dataset_id
            )
            train = []
        else:
            classes = []
            train, query = [], []

        gallery = []

        super().__init__(train, query, gallery, mode=mode, **kwargs)

        self.classes = classes


    @staticmethod
    def load_annotation(data_dir, filter_classes=None, dataset_id=0):
        ALLOWED_EXTS = ('.jpg', '.jpeg', '.png', '.gif')
        def is_valid(filename):
            return not filename.startswith('.') and filename.lower().endswith(ALLOWED_EXTS)

        def find_classes(dir, filter_names=None):
            if filter_names:
                classes = [d.name for d in os.scandir(dir) if d.is_dir() and d.name in filter_names]
            else:
                classes = [d.name for d in os.scandir(dir) if d.is_dir()]
            classes.sort()
            class_to_idx = {classes[i]: i for i in range(len(classes))}
            return class_to_idx

        class_to_idx = find_classes(data_dir, filter_classes)

        out_data = []
        for target_class in sorted(class_to_idx.keys()):
            class_index = class_to_idx[target_class]
            target_dir = osp.join(data_dir, target_class)
            if not osp.isdir(target_dir):
                continue
            for root, _, fnames in sorted(os.walk(target_dir, followlinks=True)):
                for fname in sorted(fnames):
                    path = osp.join(root, fname)
                    if is_valid(path):
                        out_data.append((path, class_index, 0, dataset_id, '', -1, -1))\

        if not len(out_data):
            print('Failed to locate images in folder ' + data_dir + f' with extensions {ALLOWED_EXTS}')

        return out_data, class_to_idx


class MultiLabelClassification(ImageDataset):
    """Multi label classification dataset.
    """

    def __init__(self, root='', mode='train', dataset_id=0, load_masks=False, **kwargs):
        if load_masks:
            raise NotImplementedError

        self.root = osp.abspath(osp.expanduser(root))
        self.data_dir = osp.dirname(self.root)
        self.annot = self.root

        required_files = [
            self.data_dir, self.annot
        ]
        self.check_before_run(required_files)

        if mode == 'train':
            train = self.load_annotation(
                self.annot,
                self.data_dir,
                dataset_id=dataset_id
            )
        else:
            train = []

        if mode == 'query':
            query = self.load_annotation(
                self.annot,
                self.data_dir,
                dataset_id=dataset_id
            )
        else:
            query = []

        gallery = []
        super(MultiLabelClassification, self).__init__(train, query, gallery, mode=mode, **kwargs)

    @staticmethod
    def load_annotation(annot_path, data_dir, dataset_id=0):
        out_data = []
        with open(annot_path) as f:
            annotation = json.load(f)
            classes = sorted(annotation['classes'])
            class_to_idx = {classes[i]: i for i in range(len(classes))}
            images_info = annotation['images']
            for img_info in images_info:
                rel_image_path, img_labels = img_info
                full_image_path = osp.join(data_dir, rel_image_path)
                labels_idx = [class_to_idx[lbl] for lbl in img_labels if lbl in class_to_idx]
                if not full_image_path or not labels_idx:
                    continue
                assert full_image_path and labels_idx
                out_data.append((full_image_path, tuple(labels_idx), 0, dataset_id, '', -1, -1))
        return out_data
