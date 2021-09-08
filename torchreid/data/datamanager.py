from __future__ import absolute_import, division, print_function

import numpy as np
import torch

from torchreid.data.datasets import init_image_dataset
from torchreid.data.sampler import build_train_sampler
from torchreid.data.transforms import build_transforms
from torchreid.utils import worker_init_fn

class DataManager(object):
    r"""Base data manager.

    Args:
        sources (str or list): source dataset(s).
        targets (str or list, optional): target dataset(s). If not given,
            it equals to ``sources``.
        height (int, optional): target image height. Default is 256.
        width (int, optional): target image width. Default is 128.
        transforms (str or list of str, optional): transformations applied to model training.
            Default is 'random_flip'.
        norm_mean (list or None, optional): data mean. Default is None (use imagenet mean).
        norm_std (list or None, optional): data std. Default is None (use imagenet std).
        use_gpu (bool, optional): use gpu. Default is True.
    """

    def __init__(
        self,
        sources=None,
        targets=None,
        height=256,
        width=128,
        transforms='random_flip',
        norm_mean=None,
        norm_std=None,
        use_gpu=False,
        apply_masks_to_test=False
    ):
        self.source_groups = sources
        if self.source_groups is None:
            raise ValueError('sources must not be None')
        if isinstance(self.source_groups, str):
            self.source_groups = [[self.source_groups]]
        elif isinstance(self.source_groups, (list, tuple)):
            self.source_groups = [[v] if isinstance(v, str) else v for v in self.source_groups]
        self.sources = [s for group in self.source_groups for s in group]

        self.targets = targets
        if self.targets is None:
            self.targets = self.sources
        if isinstance(self.targets, str):
            self.targets = [self.targets]

        self.height = height
        self.width = width
        self.transform_tr, self.transform_te = build_transforms(
            self.height,
            self.width,
            transforms=transforms,
            norm_mean=norm_mean,
            norm_std=norm_std,
            apply_masks_to_test=apply_masks_to_test,
        )

        self.use_gpu = (torch.cuda.is_available() and use_gpu)

    @property
    def num_train_pids(self):
        """Returns the number of training person identities."""
        return self._num_train_pids

    @property
    def num_train_cams(self):
        """Returns the number of training cameras."""
        return self._num_train_cams

    @property
    def data_counts(self):
        """Returns the number of samples for each ID."""
        return self._data_counts

    def return_query_and_gallery_by_name(self, name):
        """Returns query and gallery of a test dataset, each containing
        tuples of (img_path(s), pid, camid).

        Args:
            name (str): dataset name.
        """
        query_loader = self.test_dataset[name]['query']
        gallery_loader = self.test_dataset[name]['gallery']
        return query_loader, gallery_loader

    def preprocess_pil_img(self, img):
        """Transforms a PIL image to torch tensor for testing."""
        return self.transform_te(img)

    @staticmethod
    def build_dataset_map(groups):
        out_data = dict()
        for group_id, group in enumerate(groups):
            for name in group:
                out_data[name] = group_id

        return out_data

    @staticmethod
    def to_ordered_list(dict_data):
        keys = sorted(dict_data.keys())
        out_data = [dict_data[key] for key in keys]

        return out_data


class ImageDataManager(DataManager):
    r"""Image data manager.

    Args:
        root (str): root path to datasets.
        sources (str or list): source dataset(s).
        targets (str or list, optional): target dataset(s). If not given,
            it equals to ``sources``.
        height (int, optional): target image height. Default is 256.
        width (int, optional): target image width. Default is 128.
        transforms (str or list of str, optional): transformations applied to model training.
            Default is 'random_flip'.
        norm_mean (list or None, optional): data mean. Default is None (use imagenet mean).
        norm_std (list or None, optional): data std. Default is None (use imagenet std).
        use_gpu (bool, optional): use gpu. Default is True.
        combineall (bool, optional): combine train, query and gallery in a dataset for
            training. Default is False.
        batch_size_train (int, optional): number of images in a training batch. Default is 32.
        batch_size_test (int, optional): number of images in a test batch. Default is 32.
        workers (int, optional): number of workers. Default is 4.
        batch_num_instances (int, optional): number of instances per identity in a batch.
            Default is 4.
        train_sampler (str, optional): sampler. Default is RandomSampler.
        cuhk03_labeled (bool, optional): use cuhk03 labeled images.
            Default is False (defaul is to use detected images).
        cuhk03_classic_split (bool, optional): use the classic split in cuhk03.
            Default is False.
        market1501_500k (bool, optional): add 500K distractors to the gallery
            set in market1501. Default is False.

    Examples::

        datamanager = torchreid.data.ImageDataManager(
            root='path/to/reid-data',
            sources='market1501',
            height=256,
            width=128,
            batch_size_train=32,
            batch_size_test=100
        )

        # return train loader of source data
        train_loader = datamanager.train_loader

        # return test loader of target data
        test_loader = datamanager.test_loader
    """

    def __init__(
        self,
        root='',
        sources=None,
        targets=None,
        height=256,
        width=128,
        enable_masks=False,
        transforms='random_flip',
        norm_mean=None,
        norm_std=None,
        use_gpu=True,
        split_id=0,
        combineall=False,
        batch_size_train=32,
        batch_size_test=32,
        workers=4,
        train_sampler='RandomSampler',
        batch_num_instances=4,
        epoch_num_instances=-1,
        fill_instances=False,
        cuhk03_labeled=False,
        cuhk03_classic_split=False,
        market1501_500k=False,
        custom_dataset_names=[''],
        custom_dataset_roots=[''],
        custom_dataset_types=[''],
        apply_masks_to_test=False,
        min_samples_per_id=0,
        num_sampled_packages=1,
        filter_classes=None
    ):

        super(ImageDataManager, self).__init__(
            sources=sources,
            targets=targets,
            height=height,
            width=width,
            transforms=transforms,
            norm_mean=norm_mean,
            norm_std=norm_std,
            use_gpu=use_gpu,
            apply_masks_to_test=apply_masks_to_test
        )

        print('=> Loading train (source) dataset')
        train_dataset_ids_map = self.build_dataset_map(self.source_groups)

        train_dataset = []
        for name in self.sources:
            train_dataset.append(init_image_dataset(
                name,
                transform=self.transform_tr,
                mode='train',
                combineall=combineall,
                root=root,
                split_id=split_id,
                load_masks=enable_masks,
                dataset_id=train_dataset_ids_map[name],
                cuhk03_labeled=cuhk03_labeled,
                cuhk03_classic_split=cuhk03_classic_split,
                market1501_500k=market1501_500k,
                custom_dataset_names=custom_dataset_names,
                custom_dataset_roots=custom_dataset_roots,
                custom_dataset_types=custom_dataset_types,
                min_id_samples=min_samples_per_id,
                num_sampled_packages=num_sampled_packages,
                filter_classes=filter_classes
            ))
        train_dataset = sum(train_dataset)

        self._data_counts = self.to_ordered_list(train_dataset.data_counts)
        self._num_train_pids = self.to_ordered_list(train_dataset.num_train_pids)
        self._num_train_cams = self.to_ordered_list(train_dataset.num_train_cams)
        assert isinstance(self._num_train_pids, list)
        assert isinstance(self._num_train_cams, list)
        assert len(self._num_train_pids) == len(self._num_train_cams)

        self.train_loader = torch.utils.data.DataLoader(
            train_dataset,
            sampler=build_train_sampler(
                train_dataset.train,
                train_sampler,
                batch_size=max(1, min(batch_size_train, len(train_dataset))) ,
                batch_num_instances=batch_num_instances,
                epoch_num_instances=epoch_num_instances,
                fill_instances=fill_instances,
            ),
            batch_size=batch_size_train,
            shuffle=False,
            worker_init_fn=worker_init_fn,
            num_workers=workers,
            pin_memory=self.use_gpu,
            drop_last=True
        )

        print('=> Loading test (target) dataset')
        self.test_loader = {name: {'query': None, 'gallery': None} for name in self.targets}
        self.test_dataset = {name: {'query': None, 'gallery': None} for name in self.targets}

        for name in self.targets:
            if name == 'lfw':
                lfw_data = init_image_dataset(
                    name,
                    transform=self.transform_te,
                    root=root,
                )
                self.test_loader[name]['pairs'] = torch.utils.data.DataLoader(
                    lfw_data,
                    batch_size=max(min(batch_size_test, len(lfw_data)), 1),
                    shuffle=False,
                    num_workers=workers,
                    pin_memory=self.use_gpu,
                    worker_init_fn=worker_init_fn,
                    drop_last=False
                )
            else:
                # build query loader
                query_dataset = init_image_dataset(
                    name,
                    transform=self.transform_te,
                    mode='query',
                    combineall=combineall,
                    root=root,
                    split_id=split_id,
                    cuhk03_labeled=cuhk03_labeled,
                    cuhk03_classic_split=cuhk03_classic_split,
                    market1501_500k=market1501_500k,
                    custom_dataset_names=custom_dataset_names,
                    custom_dataset_roots=custom_dataset_roots,
                    custom_dataset_types=custom_dataset_types,
                    filter_classes=filter_classes
                )
                self.test_loader[name]['query'] = torch.utils.data.DataLoader(
                    query_dataset,
                    batch_size=max(min(batch_size_test, len(query_dataset)), 1),
                    shuffle=False,
                    num_workers=workers,
                    worker_init_fn=worker_init_fn,
                    pin_memory=self.use_gpu,
                    drop_last=False
                )

                # build gallery loader
                gallery_dataset = init_image_dataset(
                    name,
                    transform=self.transform_te,
                    mode='gallery',
                    combineall=combineall,
                    verbose=False,
                    root=root,
                    split_id=split_id,
                    cuhk03_labeled=cuhk03_labeled,
                    cuhk03_classic_split=cuhk03_classic_split,
                    market1501_500k=market1501_500k,
                    custom_dataset_names=custom_dataset_names,
                    custom_dataset_roots=custom_dataset_roots,
                    custom_dataset_types=custom_dataset_types,
                    filter_classes=filter_classes
                )
                self.test_loader[name]['gallery'] = torch.utils.data.DataLoader(
                    gallery_dataset,
                    batch_size=max(min(batch_size_test, len(gallery_dataset)), 1),
                    worker_init_fn=worker_init_fn,
                    shuffle=False,
                    num_workers=workers,
                    pin_memory=self.use_gpu,
                    drop_last=False
                )

                self.test_dataset[name]['query'] = query_dataset.query
                self.test_dataset[name]['gallery'] = gallery_dataset.gallery

        print('\n')
        print('  **************** Summary ****************')
        print('  source            : {}'.format(self.sources))
        print('  # source datasets : {}'.format(len(self.sources)))
        print('  # source ids      : {}'.format(sum(self.num_train_pids)))
        print('  # source images   : {}'.format(len(train_dataset)))
        print('  # source cameras  : {}'.format(sum(self.num_train_cams)))
        print('  target            : {}'.format(self.targets))
        print('  *****************************************')
        print('\n')
