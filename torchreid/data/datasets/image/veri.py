from __future__ import absolute_import, division, print_function
import os.path as osp
from os import listdir

from defusedxml import lxml as etree

from ..dataset import ImageDataset

COLORS_MAP = {
    1: 4,   # yellow
    2: 6,   # orange
    3: 11,  # green
    4: 1,   # gray
    5: 7,   # red
    6: 10,  # blue
    7: 0,   # white
    8: 5,   # golden
    9: 12,  # brown
    10: 3   # black
}

TYPES_MAP = {
    1: 0,   # sedan
    2: 1,   # suv
    3: 10,  # van
    4: 3,   # hatchback
    5: 18,  # mpv
    6: 6,   # pickup
    7: 12,  # bus
    8: 14,  # truck
    9: 5    # estate
}


class VeRi(ImageDataset):
    """VeRi-776.

    URL: `<https://github.com/VehicleReId/VeRidataset>`_

    Dataset statistics:
        - identities: 776.
        - images: 37778 (train) + 1678 (query) + 11579 (gallery).
    """
    dataset_dir = 'veri'

    def __init__(self, root='', dataset_id=0, load_masks=False, **kwargs):
        self.root = osp.abspath(osp.expanduser(root))
        self.dataset_dir = osp.join(self.root, self.dataset_dir)
        self.data_dir = self.dataset_dir

        self.train_dir = osp.join(self.data_dir, 'image_train')
        self.train_mask = osp.join(self.data_dir, 'mask_train')
        self.train_annot = osp.join(self.data_dir, 'train_label.xml')
        self.query_dir = osp.join(self.data_dir, 'image_query')
        self.gallery_dir = osp.join(self.data_dir, 'image_test')

        required_files = [
            self.data_dir, self.train_annot, self.train_dir, self.query_dir, self.gallery_dir,
        ]
        if load_masks:
            required_files.append(self.train_mask)
        self.check_before_run(required_files)

        train = self.build_annotation(
            self.train_dir, self.train_mask,
            annot=self.load_annotation(self.train_annot),
            dataset_id=dataset_id, load_masks=load_masks)
        query = self.build_annotation(
            self.query_dir, dataset_id=dataset_id)
        gallery = self.build_annotation(
            self.gallery_dir, dataset_id=dataset_id)

        train = self._compress_labels(train)

        super(VeRi, self).__init__(train, query, gallery, **kwargs)

    @staticmethod
    def load_annotation(annot_file):
        if annot_file is None or not osp.exists(annot_file):
            return None

        tree = etree.parse(annot_file)
        root = tree.getroot()

        assert len(root) == 1
        items = root[0]

        out_data = dict()
        for item in items:
            image_name = item.attrib['imageName']

            pid = int(item.attrib['vehicleID'])
            cam_id = int(item.attrib['cameraID'][1:])
            color_id = int(item.attrib['colorID'])
            type_id = int(item.attrib['typeID'])

            out_data[image_name] = dict(
                pid=pid,
                cam_id=cam_id,
                color_id=COLORS_MAP[color_id] if color_id in COLORS_MAP else -1,
                type_id=TYPES_MAP[type_id] if type_id in TYPES_MAP else -1
            )

        return out_data

    @staticmethod
    def build_annotation(images_dir, masks_dir=None, annot=None, dataset_id=0, load_masks=False):
        names = [f.replace('.jpg', '')
                 for f in listdir(images_dir)
                 if osp.isfile(osp.join(images_dir, f)) and f.endswith('.jpg')]

        if load_masks:
            mask_names = [f.replace('.png', '')
                          for f in listdir(masks_dir)
                          if osp.isfile(osp.join(masks_dir, f)) and f.endswith('.png')]
            names = list(set(names) & set(mask_names))

        data = []
        for name in names:
            name_parts = name.split('_')
            assert len(name_parts) == 4

            pid_str, cam_id_str, local_num_str, _ = name_parts

            image_name = '{}.jpg'.format(name)
            full_image_path = osp.join(images_dir, image_name)
            obj_id = int(pid_str)
            cam_id = int(cam_id_str[1:])
            assert obj_id >= 0 and cam_id >= 0

            full_mask_path = ''
            if load_masks:
                full_mask_path = osp.join(masks_dir, '{}.png'.format(name))

            if annot is None:
                color_id, type_id = -1, -1
            else:
                if image_name not in annot:
                    color_id, type_id = -1, -1
                else:
                    record = annot[image_name]
                    color_id = record['color_id']
                    type_id = record['type_id']
                    assert color_id >= 0 and type_id >= 0

            data.append((full_image_path, obj_id, cam_id, dataset_id, full_mask_path, color_id, type_id))

        return data
