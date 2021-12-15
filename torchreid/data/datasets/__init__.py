from __future__ import absolute_import, print_function
from copy import copy

from .dataset import Dataset, ImageDataset
from .image import (CUHK01, CUHK02, CUHK03, GRID, LFW, MSMT17, PRID, VRIC,
                    Classification, ClassificationImageFolder, ExternalDatasetWrapper,
                    MultiLabelClassification, CompCars, DukeMTMCreID, InternalAirport,
                    InternalCameraTampering, InternalGlobalMe, InternalMall,
                    InternalPSVIndoor, InternalPSVOutdoor, InternalSSPlatform,
                    InternalSSStreet, InternalSSTicket, InternalWildtrack,
                    Market1501, MarketTrainOnly, SenseReID, UniverseModels,
                    Vehicle1M, VeRiWild, VGGFace2, VIPeR, VMMRdb, iLIDS)

__image_datasets = {
    'market1501': Market1501,
    'cuhk03': CUHK03,
    'dukemtmcreid': DukeMTMCreID,
    'msmt17': MSMT17,
    'viper': VIPeR,
    'grid': GRID,
    'cuhk01': CUHK01,
    'ilids': iLIDS,
    'sensereid': SenseReID,
    'prid': PRID,
    'cuhk02': CUHK02,
    'vric': VRIC,
    'compcars': CompCars,
    'vmmrdb': VMMRdb,
    'vehicle1m': Vehicle1M,
    'universemodels': UniverseModels,
    'veriwild': VeRiWild,
    'market-train': MarketTrainOnly,
    'int-airport': InternalAirport,
    'int-camera-tampering': InternalCameraTampering,
    'int-globalme': InternalGlobalMe,
    'int-mall': InternalMall,
    'int-psv-indoor': InternalPSVIndoor,
    'int-psv-outdoor': InternalPSVOutdoor,
    'int-ss-platform': InternalSSPlatform,
    'int-ss-street': InternalSSStreet,
    'int-ss-ticket': InternalSSTicket,
    'int-wildtrack': InternalWildtrack,
    'vgg_face2': VGGFace2,
    'lfw': LFW,
    'classification': Classification,
    'classification_image_folder' : ClassificationImageFolder,
    'external_classification_wrapper' : ExternalDatasetWrapper,
    'multilabel_classification': MultiLabelClassification,
}


def init_image_dataset(name, custom_dataset_names=[''],
                       custom_dataset_roots=[''],
                       custom_dataset_types=[''], **kwargs):
    """Initializes an image dataset."""

    #handle also custom datasets
    avai_datasets = list(__image_datasets.keys())
    assert len(name) > 0
    if name not in avai_datasets and name not in custom_dataset_names:
        raise ValueError(
            'Invalid dataset name. Received "{}", '
            'but expected to be one of {} {}'.format(name, avai_datasets, custom_dataset_names)
        )
    if name in custom_dataset_names:
        assert len(custom_dataset_names) == len(custom_dataset_types)
        assert len(custom_dataset_names) == len(custom_dataset_roots)
        i = custom_dataset_names.index(name)
        new_kwargs = copy(kwargs)
        if custom_dataset_types[i] == 'external_classification_wrapper':
            new_kwargs['data_provider'] = custom_dataset_roots[i]
        else:
            new_kwargs['root'] = custom_dataset_roots[i]
        return __image_datasets[custom_dataset_types[i]](**new_kwargs)

    return __image_datasets[name](**kwargs)



def register_image_dataset(name, dataset):
    """Registers a new image dataset.

    Args:
        name (str): key corresponding to the new dataset.
        dataset (Dataset): the new dataset class.

    Examples::

        import torchreid
        import NewDataset
        torchreid.data.register_image_dataset('new_dataset', NewDataset)
        # single dataset case
        datamanager = torchreid.data.ImageDataManager(
            root='reid-data',
            sources='new_dataset'
        )
        # multiple dataset case
        datamanager = torchreid.data.ImageDataManager(
            root='reid-data',
            sources=['new_dataset', 'dukemtmcreid']
        )
    """
    global __image_datasets
    curr_datasets = list(__image_datasets.keys())
    if name in curr_datasets:
        raise ValueError(
            'The given name already exists, please choose '
            'another name excluding {}'.format(curr_datasets)
        )
    __image_datasets[name] = dataset
