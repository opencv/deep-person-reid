from __future__ import absolute_import, division, print_function
import copy
import operator
import os.path as osp
import tarfile
import zipfile
from collections import defaultdict

import numpy as np
import torch

from torchreid.utils import download_url, mkdir_if_missing, read_image


class Dataset:
    """An abstract class representing a Dataset.

    This is the base class for ``ImageDataset``.

    Args:
        train (list): contains tuples of (img_path(s), pid, camid).
        query (list): contains tuples of (img_path(s), pid, camid).
        gallery (list): contains tuples of (img_path(s), pid, camid).
        transform: transform function.
        mode (str): 'train', 'query' or 'gallery'.
        combineall (bool): combines train, query and gallery in a dataset for training.
        verbose (bool): show information.
    """
    _junk_pids = []  # contains useless person IDs, e.g. background, false detections

    def __init__(
        self,
        train,
        query,
        gallery,
        transform=None,
        mode='train',
        combineall=False,
        verbose=True,
        min_id_samples=0,
        num_sampled_packages=1,
        **kwargs
    ):
        self.train = train
        self.query = query
        self.gallery = gallery
        self.transform = transform
        self.mode = mode
        self.combineall = combineall
        self.verbose = verbose

        self.num_sampled_packages = num_sampled_packages
        assert self.num_sampled_packages >= 1

        self.num_train_pids = self.get_num_pids(self.train)
        self.num_train_cams = self.get_num_cams(self.train)
        if len(self.num_train_pids) == 0: # workaround for classification: query is a validation set
            self.num_train_pids = self.get_num_pids(self.query)

        if self.combineall:
            self.combine_all()

        self._cut_train_ids(min_id_samples)
        self.data_counts = self.get_data_counts(self.train)

        if self.mode == 'train':
            self.data = self.train
        elif self.mode == 'query':
            self.data = self.query
        elif self.mode == 'gallery':
            self.data = self.gallery
        else:
            raise ValueError('Invalid mode. Got {}, but expected to be '
                             'one of [train | query | gallery]'.format(self.mode))

        if self.verbose:
            self.show_summary()

    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        return len(self.data)

    def __add__(self, other):
        """Adds two datasets together (only the train set)."""
        updated_train = copy.deepcopy(self.train)

        for record in other.train:
            dataset_id = record[3] if len(record) > 3 else 0

            num_train_pids = 0
            if dataset_id in self.num_train_pids:
                num_train_pids = self.num_train_pids[dataset_id]
            old_obj_id = record[1]
            new_obj_id = old_obj_id + num_train_pids

            num_train_cams = 0
            if dataset_id in self.num_train_cams:
                num_train_cams = self.num_train_cams[dataset_id]
            old_cam_id = record[2]
            new_cam_id = old_cam_id + num_train_cams

            updated_record = record[:1] + (new_obj_id, new_cam_id) + record[3:]
            updated_train.append(updated_record)

        ###################################
        # Things to do beforehand:
        # 1. set verbose=False to avoid unnecessary print
        # 2. set combineall=False because combineall would have been applied
        #    if it was True for a specific dataset, setting it to True will
        #    create new IDs that should have been included
        ###################################

        return ImageDataset(
            updated_train,
            self.query,
            self.gallery,
            transform=self.transform,
            mode=self.mode,
            combineall=False,
            verbose=False
        )

    def __radd__(self, other):
        """Supports sum([dataset1, dataset2, dataset3])."""
        if other == 0:
            return self
        else:
            return self.__add__(other)

    @staticmethod
    def parse_data(data):
        """Parses data list and returns the number of person IDs
        and the number of camera views.

        Args:
            data (list): contains tuples of (img_path(s), pid, camid)
        """
        pids, cams = defaultdict(set), defaultdict(set)
        for record in data:
            dataset_id = record[3] if len(record) > 3 else 0
            if isinstance(record[1], (tuple, list)):
                for ids in record[1]:
                    pids[dataset_id].add(ids)
            else:
                pids[dataset_id].add(record[1])
            cams[dataset_id].add(record[2])
        num_pids = {dataset_id: len(dataset_pids) for dataset_id, dataset_pids in pids.items()}
        num_cams = {dataset_id: len(dataset_cams) for dataset_id, dataset_cams in cams.items()}

        return num_pids, num_cams

    def get_num_pids(self, data):
        """Returns the number of training person identities."""
        return self.parse_data(data)[0]

    def get_num_cams(self, data):
        """Returns the number of training cameras."""
        return self.parse_data(data)[1]

    @staticmethod
    def get_data_counts(data):
        counts = dict()
        for record in data:
            dataset_id = record[3] if len(record) > 3 else 0
            if dataset_id not in counts:
                counts[dataset_id] = defaultdict(int)

            obj_id = record[1]
            counts[dataset_id][obj_id] += 1

        return counts

    def get_record(self, index):
        return self.data[index]

    def show_summary(self):
        """Shows dataset statistics."""
        pass

    @staticmethod
    def _get_obj_ids(data, junk_obj_ids, obj_ids=None):
        if obj_ids is None:
            obj_ids = defaultdict(set)

        for record in data:
            obj_id = record[1]
            if obj_id in junk_obj_ids:
                continue

            dataset_id = record[3] if len(record) > 3 else 0
            obj_ids[dataset_id].add(obj_id)

        return obj_ids

    @staticmethod
    def _relabel(data, junk_obj_ids, id2label_map, num_train_ids):
        out_data = []
        for record in data:
            obj_id = record[1]
            if obj_id in junk_obj_ids:
                continue

            dataset_id = record[3] if len(record) > 3 else 0
            ids_shift = num_train_ids[dataset_id] if dataset_id in num_train_ids else 0
            updated_obj_id = id2label_map[dataset_id][obj_id] + ids_shift

            updated_record = record[:1] + (updated_obj_id,) + record[2:]
            out_data.append(updated_record)

        return out_data

    def _cut_train_ids(self, min_imgs):
        if min_imgs < 2:
            return

        self.train = sorted(self.train, key=operator.itemgetter(1))

        id_counters = {}
        for path, pid, cam in self.train:
            if pid in id_counters:
                id_counters[pid] += 1
            else:
                id_counters[pid] = 1
        pids_to_del = set()
        for k in id_counters:
            if id_counters[k] < min_imgs:
                pids_to_del.add(k)

        filtered_train = []
        removed_pids = set()
        for path, pid, cam in self.train:
            if pid in pids_to_del:
                removed_pids.add(pid)
            else:
                filtered_train.append((path, pid - len(removed_pids), cam))

        self.train = filtered_train

        self.num_train_pids = self.get_num_pids(self.train)
        self.num_train_cams = self.get_num_cams(self.train)

    def combine_all(self):
        """Combines train, query and gallery in a dataset for training."""
        combined = copy.deepcopy(self.train)

        new_obj_ids = self._get_obj_ids(self.query, self._junk_pids)
        new_obj_ids = self._get_obj_ids(self.gallery, self._junk_pids, new_obj_ids)

        id2label_map = dict()
        for dataset_id, dataset_ids in new_obj_ids.items():
            id2label_map[dataset_id] = {obj_id: i for i, obj_id in enumerate(set(dataset_ids))}

        combined += self._relabel(self.query, self._junk_pids, id2label_map, self.num_train_pids)
        combined += self._relabel(self.gallery, self._junk_pids, id2label_map, self.num_train_pids)

        self.train = combined
        self.num_train_pids = self.get_num_pids(self.train)

    def download_dataset(self, dataset_dir, dataset_url):
        """Downloads and extracts dataset.

        Args:
            dataset_dir (str): dataset directory.
            dataset_url (str): url to download dataset.
        """
        if osp.exists(dataset_dir):
            return

        if dataset_url is None:
            raise RuntimeError(
                '{} dataset needs to be manually '
                'prepared, please follow the '
                'document to prepare this dataset'.format(
                    self.__class__.__name__
                )
            )

        print('Creating directory "{}"'.format(dataset_dir))
        mkdir_if_missing(dataset_dir)
        fpath = osp.join(dataset_dir, osp.basename(dataset_url))

        print(
            'Downloading {} dataset to "{}"'.format(
                self.__class__.__name__, dataset_dir
            )
        )
        download_url(dataset_url, fpath)

        print('Extracting "{}"'.format(fpath))
        try:
            tar = tarfile.open(fpath)
            tar.extractall(path=dataset_dir)
            tar.close()
        except:
            zip_ref = zipfile.ZipFile(fpath, 'r')
            zip_ref.extractall(dataset_dir)
            zip_ref.close()

        print('{} dataset is ready'.format(self.__class__.__name__))

    @staticmethod
    def check_before_run(required_files):
        """Checks if required files exist before going deeper.

        Args:
            required_files (str or list): string file name(s).
        """
        if isinstance(required_files, str):
            required_files = [required_files]

        for fpath in required_files:
            if not osp.exists(fpath):
                raise RuntimeError('"{}" is not found'.format(fpath))

    @staticmethod
    def _compress_labels(data):
        if len(data) == 0:
            return data

        pid_container = set(record[1] for record in data)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}

        out_data = [record[:1] + (pid2label[record[1]],) + record[2:] for record in data]

        return out_data

    def __repr__(self):
        num_train_pids, num_train_cams = self.parse_data(self.train)
        num_query_pids, num_query_cams = self.parse_data(self.query)
        num_gallery_pids, num_gallery_cams = self.parse_data(self.gallery)

        msg = '  ----------------------------------------\n' \
              '  subset   | # ids | # items | # cameras\n' \
              '  ----------------------------------------\n' \
              '  train    | {:5d} | {:7d} | {:9d}\n' \
              '  query    | {:5d} | {:7d} | {:9d}\n' \
              '  gallery  | {:5d} | {:7d} | {:9d}\n' \
              '  ----------------------------------------\n' \
              '  items: images/tracklets for image dataset\n'.format(
                  sum(num_train_pids.values()), len(self.train), sum(num_train_cams.values()),
                  sum(num_query_pids.values()), len(self.query), sum(num_query_cams.values()),
                  sum(num_gallery_pids.values()), len(self.gallery), sum(num_gallery_cams.values())
              )

        return msg


class ImageDataset(Dataset):
    """A base class representing ImageDataset.

    All other image datasets should subclass it.

    ``__getitem__`` returns an image given index.
    It will return ``img``, ``pid``, ``camid`` and ``img_path``
    where ``img`` has shape (channel, height, width). As a result,
    data in each batch has shape (batch_size, channel, height, width).
    """

    def __init__(self, train, query, gallery, **kwargs):
        super(ImageDataset, self).__init__(train, query, gallery, **kwargs)
        self.classes = {}

    def __getitem__(self, index):
        input_record = self.data[index]

        image = read_image(input_record[0], grayscale=False)
        obj_id = input_record[1]
        cam_id = input_record[2]

        if len(input_record) > 3:
            dataset_id = input_record[3]
            if isinstance(obj_id, (tuple, list)): # when multi-label classification is available
                targets = torch.zeros(self.num_train_pids[dataset_id])
                for obj in obj_id:
                    targets[obj] = 1
                obj_id = targets

            mask = ''
            if input_record[4] != '':
                mask = read_image(input_record[4], grayscale=True)

            if self.mode == 'train' and self.num_sampled_packages > 1:
                assert self.transform is not None

                transformed_image, transformed_mask = [], []
                for _ in range(self.num_sampled_packages):
                    gen_image, gen_mask = self.transform((image, mask))

                    transformed_image.append(gen_image)
                    transformed_mask.append(gen_mask)

                transformed_image = torch.stack(transformed_image)
                transformed_mask = torch.stack(transformed_mask) if mask != '' else ''
            elif self.transform is not None:
                transformed_image, transformed_mask = self.transform((image, mask))
            else:
                transformed_image, transformed_mask = image, mask

            output_record = (transformed_image, obj_id, cam_id, dataset_id, transformed_mask) + input_record[5:]
        else:
            if self.mode == 'train' and self.num_sampled_packages > 1:
                assert self.transform is not None

                transformed_image = []
                for _ in range(self.num_sampled_packages):
                    gen_image, _ = self.transform((image, ''))

                    transformed_image.append(gen_image)

                transformed_image = torch.stack(transformed_image)
            elif self.transform is not None:
                transformed_image, _ = self.transform((image, ''))
            else:
                transformed_image = image

            output_record = transformed_image, obj_id, cam_id

        return output_record

    def show_summary(self):
        num_train_pids, num_train_cams = self.parse_data(self.train)
        num_query_pids, num_query_cams = self.parse_data(self.query)
        num_gallery_pids, num_gallery_cams = self.parse_data(self.gallery)

        print('=> Loaded {}'.format(self.__class__.__name__))
        print('  ----------------------------------------')
        print('  subset   | # ids | # images | # cameras')
        print('  ----------------------------------------')
        print('  train    | {:5d} | {:8d} | {:9d}'.format(
            sum(num_train_pids.values()), len(self.train), sum(num_train_cams.values())))
        print('  query    | {:5d} | {:8d} | {:9d}'.format(
            sum(num_query_pids.values()), len(self.query), sum(num_query_cams.values())))
        print('  gallery  | {:5d} | {:8d} | {:9d}'.format(
            sum(num_gallery_pids.values()), len(self.gallery), sum(num_gallery_cams.values())))
        print('  ----------------------------------------')
