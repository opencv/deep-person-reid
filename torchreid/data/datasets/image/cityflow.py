from __future__ import absolute_import, division, print_function
import os.path as osp

from defusedxml import lxml as etree

from ..dataset import ImageDataset


class CityFlow(ImageDataset):
    """CityFlow.

    URL: `<https://www.aicitychallenge.org/2020-track2-download>`_

    Dataset statistics:
        - identities: 666.
        - images: 36935 (train) + 1052 (query) + 18290 (gallery).
    """
    dataset_dir = 'cityflow_custom_test'

    def __init__(self, root='', dataset_id=0, load_masks=False, **kwargs):
        self.root = osp.abspath(osp.expanduser(root))
        self.dataset_dir = osp.join(self.root, self.dataset_dir)
        self.data_dir = self.dataset_dir

        self.train_dir = osp.join(self.data_dir, 'image_train')
        self.query_dir = osp.join(self.data_dir, 'image_query')
        self.gallery_dir = osp.join(self.data_dir, 'image_test')

        self.train_annot = osp.join(self.data_dir, 'train_label.xml')
        self.query_annot = osp.join(self.data_dir, 'query_label.xml')
        self.gallery_annot = osp.join(self.data_dir, 'test_label.xml')

        required_files = [
            self.data_dir, self.query_dir, self.gallery_dir,
            self.query_annot, self.gallery_annot,
            self.train_dir, self.train_annot
        ]
        self.check_before_run(required_files)

        train = self.load_annotation(
            self.train_annot,
            self.train_dir,
            dataset_id=dataset_id,
            load_masks=load_masks
        )
        query = self.load_annotation(self.query_annot, self.query_dir)
        gallery = self.load_annotation(self.gallery_annot, self.gallery_dir)

        train = self._compress_labels(train)

        super(CityFlow, self).__init__(train, query, gallery, **kwargs)

    @staticmethod
    def load_annotation(annot_path, data_dir, dataset_id=0, load_masks=False):
        if load_masks:
            raise NotImplementedError

        tree = etree.parse(annot_path)
        root = tree.getroot()

        assert len(root) == 1
        items = root[0]

        out_data = list()
        for item in items:
            image_name = item.attrib['imageName']
            full_image_path = osp.join(data_dir, image_name)
            assert osp.exists(full_image_path)

            obj_id = int(item.attrib['vehicleID'])
            cam_id = int(item.attrib['cameraID'][1:])

            color_id = int(item.attrib['colorID']) if 'colorID' in item.attrib else -1
            type_id = int(item.attrib['typeID']) if 'typeID' in item.attrib else -1

            out_data.append((full_image_path, obj_id, cam_id, dataset_id, '', color_id, type_id))

        return out_data
