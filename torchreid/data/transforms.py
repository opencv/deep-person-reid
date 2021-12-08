from __future__ import absolute_import, division, print_function
import math
import random
from collections import deque
from os.path import exists, join

import cv2
import numpy as np
import torch
import re
from PIL import Image, ImageOps, ImageEnhance, ImageDraw
from torchvision.transforms import (ColorJitter, Compose, Normalize,
                                    ToTensor)
from torchvision.transforms import RandomCrop as TorchRandomCrop
from torchvision.transforms import functional as F
from randaugment import RandAugment

from torchreid.utils.tools import read_image
from ..data.datasets.image.lfw import FivePointsAligner


class RandomHorizontalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, input_tuple):
        if random.random() > self.p:
            return input_tuple

        img, mask = input_tuple

        img = F.hflip(img)
        mask = F.hflip(mask) if mask != '' else mask

        return img, mask


class CenterCrop(object):
    def __init__(self, margin=0, **kwargs):
        self.margin = margin

    def __call__(self, input_tuple, *args, **kwargs):
        if self.margin <= 0:
            return input_tuple

        img, mask = input_tuple
        img_height, img_width, _ = img.shape
        if img_width - 2 * self.margin < 2 or img_height - 2 * self.margin < 2:
            return input_tuple

        img = img[self.margin : img_height - self.margin, self.margin : img_width - self.margin]
        mask = mask[self.margin : img_height - self.margin,
                    self.margin : img_width - self.margin] if mask != '' else mask

        return img, mask


class RandomCrop(object):
    def __init__(self, p=0.5, scale=0.9, static=False, margin=None,
                    target_ar=None, align_ar=False, align_center=False, **kwargs):
        self.p = p
        assert 0.0 <= self.p <= 1.0
        self.scale = scale
        assert 0.0 < self.scale < 1.0
        self.margin = margin
        self.target_ar = target_ar
        self.align_center = align_center
        self.static = static
        self.align_ar = align_ar
        if self.align_ar:
            assert self.target_ar is not None and self.target_ar > 0

    def __call__(self, input_tuple, *args, **kwargs):
        if random.uniform(0, 1) > self.p:
            return input_tuple

        img, mask = input_tuple
        img_height, img_width, _ = img.shape

        if self.align_ar:
            source_ar = float(img_height) / float(img_width)
            target_ar = random.uniform(min(source_ar, self.target_ar), max(source_ar, self.target_ar))

            if target_ar < source_ar:
                max_crop_width = img_width
                max_crop_height = target_ar * max_crop_width
            else:
                max_crop_height = img_height
                max_crop_width = max_crop_height / target_ar
        else:
            max_crop_width = img_width
            max_crop_height = img_height

        if self.margin is None or self.margin <= 0:
            min_scale = self.scale
        else:
            width_rest = max(1, img_width - 2 * self.margin)
            height_rest = max(1, img_height - 2 * self.margin)

            min_width_scale = float(width_rest) / float(img_width)
            min_height_scale = float(height_rest) / float(img_height)
            min_scale = max(min_width_scale, min_height_scale)

        if self.static:
            scale = min_scale
        else:
            scale = random.uniform(min_scale, 1.0)
        crop_width = int(round(scale * max_crop_width))
        crop_height = int(round(scale * max_crop_height))

        if self.align_center:
            min_crop_width = min_scale * img_width
            min_crop_height = min_scale * img_height

            center_x = 0.5 * img_width
            center_y = 0.5 * img_height

            x_shift_range = (max(0, center_x + 0.5 * min_crop_width - crop_width),
                             min(img_width - crop_width, center_x - 0.5 * min_crop_width))
            y_shift_range = (max(0, center_y + 0.5 * min_crop_height - crop_height),
                             min(img_height - crop_height, center_y - 0.5 * min_crop_height))
        else:
            x_shift_range = 0, img_width - crop_width
            y_shift_range = 0, img_height - crop_height

        x1 = int(round(random.uniform(*x_shift_range)))
        y1 = int(round(random.uniform(*y_shift_range)))

        img = img[y1 : y1 + crop_height, x1 : x1 + crop_width]
        mask = mask[y1 : y1 + crop_height, x1 : x1 + crop_width] if mask != '' else mask
        return img, mask


class RandomErasing(object):
    """Randomly erases an image patch.

    Origin: `<https://github.com/zhunzhong07/Random-Erasing>`_

    Reference:
        Zhong et al. Random Erasing Data Augmentation.

    Args:
        probability (float, optional): probability that this operation takes place.
            Default is 0.5.
        sl (float, optional): min erasing area.
        sh (float, optional): max erasing area.
        r1 (float, optional): min aspect ratio.
    """

    def __init__(self, p=0.5, sl=0.02, sh=0.4, rl=0.5, rh=2.0, fill_color=None, norm_image=True, **kwargs):
        self.probability = p
        self.sl = sl
        self.sh = sh
        self.rl = rl
        self.rh = rh
        self.norm_image = norm_image
        self.fill_color = fill_color
        if self.fill_color is not None and len(self.fill_color) in (1, 3):
            if len(self.fill_color) == 1:
                self.fill_color = [self.fill_color] * 3
        else:
            self.fill_color = None

    def __call__(self, input_tuple, *args, **kwargs):
        if random.uniform(0, 1) > self.probability:
            return input_tuple

        img, mask = input_tuple
        img_size = img.size() if self.norm_image else img.size

        for attempt in range(100):
            source_area = img_size[0] * img_size[1]
            target_area = random.uniform(self.sl, self.sh) * source_area
            aspect_ratio = random.uniform(self.rl, self.rh)

            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))

            if w < img_size[1] and h < img_size[0]:
                x1 = random.randint(0, img_size[0] - h)
                y1 = random.randint(0, img_size[1] - w)

                fill_color = self.fill_color if self.fill_color is not None else [random.randint(0, 255)] * 3
                if self.norm_image:
                    fill_color = np.array(fill_color) / 255.0

                img = img if self.norm_image else np.array(img)
                img[x1:x1 + h, y1:y1 + w, 0] = fill_color[0]
                img[x1:x1 + h, y1:y1 + w, 1] = fill_color[1]
                img[x1:x1 + h, y1:y1 + w, 2] = fill_color[2]
                img = img if self.norm_image else Image.fromarray(img)

                return img, mask

        return img, mask


class ColorAugmentation(object):
    """Randomly alters the intensities of RGB channels.

    Reference:
        Krizhevsky et al. ImageNet Classification with Deep ConvolutionalNeural
        Networks. NIPS 2012.

    Args:
        p (float, optional): probability that this operation takes place.
            Default is 0.5.
    """

    def __init__(self, p=0.5, **kwargs):
        self.p = p
        self.eig_vec = torch.Tensor([[0.4009, 0.7192, -0.5675],
                                     [-0.8140, -0.0045, -0.5808],
                                     [0.4203, -0.6948, -0.5836],
                                     ])
        self.eig_val = torch.Tensor([[0.2175, 0.0188, 0.0045]])

    def __call__(self, input_tuple, *args, **kwargs):
        if random.uniform(0, 1) > self.p:
            return input_tuple

        tensor, mask = input_tuple

        alpha = torch.normal(mean=torch.zeros_like(self.eig_val)) * 0.1
        quatity = torch.mm(self.eig_val * alpha, self.eig_vec)
        tensor = tensor + quatity.view(3, 1, 1)

        return tensor, mask


class RandomPatch(object):
    """Random patch data augmentation.

    There is a patch pool that stores randomly extracted pathces from person images.

    For each input image, RandomPatch
        1) extracts a random patch and stores the patch in the patch pool;
        2) randomly selects a patch from the patch pool and pastes it on the
           input (at random position) to simulate occlusion.

    Reference:
        - Zhou et al. Omni-Scale Feature Learning for Person Re-Identification. ICCV, 2019.
        - Zhou et al. Learning Generalisable Omni-Scale Representations
          for Person Re-Identification. arXiv preprint, 2019.
    """

    def __init__(self, p=0.5, pool_capacity=50000, min_sample_size=100,
                 patch_min_area=0.01, patch_max_area=0.5, patch_min_ratio=0.1,
                 prob_rotate=0.5, prob_flip_leftright=0.5, **kwargs):
        self.prob_happen = p

        self.patch_min_area = patch_min_area
        self.patch_max_area = patch_max_area
        self.patch_min_ratio = patch_min_ratio

        self.prob_rotate = prob_rotate
        self.prob_flip_leftright = prob_flip_leftright

        self.patchpool = deque(maxlen=pool_capacity)
        self.min_sample_size = min_sample_size

    def generate_wh(self, W, H):
        area = W * H
        for attempt in range(100):
            target_area = random.uniform(self.patch_min_area, self.patch_max_area) * area
            aspect_ratio = random.uniform(self.patch_min_ratio, 1.0 / self.patch_min_ratio)
            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))
            if w < W and h < H:
                return w, h

        return None, None

    def transform_patch(self, patch):
        if random.uniform(0, 1) > self.prob_flip_leftright:
            patch = patch.transpose(Image.FLIP_LEFT_RIGHT)
        if random.uniform(0, 1) > self.prob_rotate:
            patch = patch.rotate(random.randint(-10, 10))
        return patch

    def __call__(self, input_tuple):
        img, mask = input_tuple

        W, H = img.size  # original image size

        # collect new patch
        w, h = self.generate_wh(W, H)
        if w is not None and h is not None:
            x1 = random.randint(0, W - w)
            y1 = random.randint(0, H - h)
            new_patch = img.crop((x1, y1, x1 + w, y1 + h))
            self.patchpool.append(new_patch)

        if len(self.patchpool) < self.min_sample_size:
            return img, mask

        if random.uniform(0, 1) > self.prob_happen:
            return img, mask

        # paste a randomly selected patch on a random position
        patch = random.sample(self.patchpool, 1)[0]
        patchW, patchH = patch.size
        x1 = random.randint(0, W - patchW)
        y1 = random.randint(0, H - patchH)
        patch = self.transform_patch(patch)
        img.paste(patch, (x1, y1))

        return img, mask


class RandomAugment(RandAugment):
    def __init__(self, p=1., **kwargs):
        self.p = p
        super().__init__()

    def __call__(self, input_tuple):
        if random.uniform(0, 1) > self.p:
            return input_tuple

        img, mask = input_tuple
        img = super().__call__(img)

        return img, mask

class RandomColorJitter(ColorJitter):
    def __init__(self, p=0.5, brightness=0.2, contrast=0.15, saturation=0, hue=0, **kwargs):
        self.p = p
        super().__init__(brightness=brightness, contrast=contrast, saturation=saturation, hue=hue)

    def __call__(self, input_tuple):
        if random.uniform(0, 1) > self.p:
            return input_tuple

        img, mask = input_tuple
        img = self.forward(img)

        return img, mask


class RandomGrayscale(object):
    def __init__(self, p=0.1):
        self.p = p

    def __call__(self, input_tuple):
        if random.uniform(0, 1) > self.p:
            return input_tuple

        img, mask = input_tuple

        num_output_channels = 1 if img.mode == 'L' else 3
        img = F.to_grayscale(img, num_output_channels=num_output_channels)

        return img, mask

class Equalize(object):
    def __init__(self, p=0.5, **kwargs):
        self.p = p

    def __call__(self, input_tuple):
        if random.uniform(0, 1) > self.p:
            return input_tuple

        img, mask = input_tuple
        img = ImageOps.equalize(img)

        return img, mask

class Posterize(object):
    def __init__(self, p=0.5, bits=1, **kwargs):
        self.p = p
        self.bits = bits

    def __call__(self, input_tuple):
        if random.uniform(0, 1) > self.p:
            return input_tuple
        bit = random.randint(self.bits, 6)

        img, mask = input_tuple
        img = ImageOps.posterize(img, bit)

        return img, mask

class RandomNegative(object):
    def __init__(self, p=0.1, **kwargs):
        self.p = p

    def __call__(self, input_tuple):
        if random.uniform(0, 1) > self.p:
            return input_tuple

        img, mask = input_tuple
        img = ImageOps.invert(img)

        return img, mask


class ForceGrayscale(object):
    def __init__(self):
        pass

    def __call__(self, input_tuple):
        img, mask = input_tuple

        num_output_channels = 1 if img.mode == 'L' else 3
        img = F.to_grayscale(img, num_output_channels=num_output_channels)

        return img, mask


class RandomRotate(object):
    """Random rotate
    """

    def __init__(self, p=0.5, angle=(-5, 5), values=None, **kwargs):
        self.p = p
        self.angle = angle

        self.discrete = values is not None and len([v for v in values if v != 0]) > 0
        self.values = values

    def __call__(self, input_tuple, *args, **kwargs):
        if random.uniform(0, 1) > self.p:
            return input_tuple

        img, mask = input_tuple

        if self.discrete:
            rnd_angle = float(self.values[random.randint(0, len(self.values) - 1)])
        else:
            rnd_angle = random.randint(self.angle[0], self.angle[1])

        img = F.rotate(img, rnd_angle, expand=False, center=None)
        if mask != '':
            rgb_mask = mask.convert('RGB')
            rgb_mask = F.rotate(rgb_mask, rnd_angle, expand=False, center=None)
            mask = rgb_mask.convert('L')

        return img, mask

class CoarseDropout(object):
    """CoarseDropout of the rectangular regions in the image.
    Args:
        max_holes (int): Maximum number of regions to zero out.
        max_height (int): Maximum height of the hole.
        max_width (int): Maximum width of the hole.
        min_holes (int): Minimum number of regions to zero out. If `None`,
            `min_holes` is be set to `max_holes`. Default: `None`.
        min_height (int): Minimum height of the hole. Default: None. If `None`,
            `min_height` is set to `max_height`. Default: `None`.
        min_width (int): Minimum width of the hole. If `None`, `min_height` is
            set to `max_width`. Default: `None`.
        fill_value (int, float, lisf of int, list of float): value for dropped pixels.
        mask_fill_value (int, float, lisf of int, list of float): fill value for dropped pixels
            in mask. If None - mask is not affected.
    Targets:
        image, mask
    """

    def __init__(
        self,
        max_holes=8,
        max_height=8,
        max_width=8,
        min_holes=None,
        min_height=None,
        min_width=None,
        fill_value=0,
        mask_fill_value=None,
        p=0.5,
        **kwargs
    ):
        self.p = p
        self.max_holes = max_holes
        self.max_height = max_height
        self.max_width = max_width
        self.min_holes = min_holes if min_holes is not None else max_holes
        self.min_height = min_height if min_height is not None else max_height
        self.min_width = min_width if min_width is not None else max_width
        self.fill_value = fill_value
        self.mask_fill_value = mask_fill_value
        if not 0 < self.min_holes <= self.max_holes:
            raise ValueError("Invalid combination of min_holes and max_holes. Got: {}".format([min_holes, max_holes]))
        if not 0 < self.min_height <= self.max_height:
            raise ValueError(
                "Invalid combination of min_height and max_height. Got: {}".format([min_height, max_height])
            )
        if not 0 < self.min_width <= self.max_width:
            raise ValueError("Invalid combination of min_width and max_width. Got: {}".format([min_width, max_width]))

    def __call__(self, input_tuple):

        if random.uniform(0, 1) > self.p:
            return input_tuple

        img, mask = input_tuple
        height, width = img.size

        holes = []
        for _n in range(random.randint(self.min_holes, self.max_holes)):
            hole_height = random.randint(self.min_height, self.max_height)
            hole_width = random.randint(self.min_width, self.max_width)

            y1 = random.randint(0, height - hole_height)
            x1 = random.randint(0, width - hole_width)
            y2 = y1 + hole_height
            x2 = x1 + hole_width
            holes.append((x1, y1, x2, y2))

        img = self.cutout(img, holes, self.fill_value)
        return img, mask

    @staticmethod
    def cutout(image, holes, fill_value):
        # Make a copy of the input image since we don't want to modify it directly
        image = np.array(image)
        for x1, y1, x2, y2 in holes:
            image[y1:y2, x1:x2] = fill_value

        return Image.fromarray(image)

class Cutout(object):
    """Randomly mask out one or more patches from an image.
    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
    """
    def __init__(self, cutout_factor=0.3, fill_color='random', p=0.5, **kwargs):
        self.p = p
        self.p = p
        self.cutout_factor = cutout_factor
        self.fill_color = fill_color

    def __call__(self, input_tuple):
        if random.uniform(0, 1) > self.p:
            return input_tuple

        img, img_mask = input_tuple
        img_draw = ImageDraw.Draw(img)
        h, w = img.size[0], img.size[1]
        h_cutout = int(self.cutout_factor * h + 0.5)
        w_cutout = int(self.cutout_factor * w + 0.5)
        y_c = np.random.randint(h)
        x_c = np.random.randint(w)

        y1 = np.clip(y_c - h_cutout // 2, 0, h)
        y2 = np.clip(y_c + h_cutout // 2, 0, h)
        x1 = np.clip(x_c - w_cutout // 2, 0, w)
        x2 = np.clip(x_c + w_cutout // 2, 0, w)
        if self.fill_color == 'random':
            fill_color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        else:
            assert isinstance(self.fill_color, (tuple, list))
            fill_color = self.fill_color
        img_draw.rectangle([x1, y1, x2, y2], fill=fill_color)

        return img, img_mask

class RandomGrid(object):
    """Random grid
    """

    def __init__(self, p=0.5, color=-1, grid_size=(24, 64), thickness=(1, 1), angle=(0, 180), **kwargs):
        self.p = p
        self.color = color
        self.grid_size = grid_size
        self.thickness = thickness
        self.angle = angle

    def __call__(self, input_tuple, *args, **kwargs):
        if random.uniform(0, 1) > self.p:
            return input_tuple

        img, mask = input_tuple

        if self.color == (-1, -1, -1):  # Random color
            color = tuple([random.randint(0, 256) for _ in range(3)])
        else:
            color = self.color

        grid_size = random.randint(*self.grid_size)
        thickness = random.randint(*self.thickness)
        angle = random.randint(*self.angle)

        return self.draw_grid(img, grid_size, color, thickness, angle), mask

    @staticmethod
    def draw_grid(img, grid_size, color, thickness, angle):
        h, w = img.shape[:2]
        mask = np.zeros((h * 8, w * 8, 3), dtype='uint8')
        mask_h, mask_w = mask.shape[:2]
        for i in range(0, mask_h, grid_size):
            p1 = (0, i)
            p2 = (mask_w, i + grid_size)
            mask = cv2.line(mask, p1, p2, (255, 255, 255), thickness)
        for i in range(0, mask_w, grid_size):
            p1 = (i, 0)
            p2 = (i + grid_size, mask_h)
            mask = cv2.line(mask, p1, p2, (255, 255, 255), thickness)

        center = (mask_w // 2, mask_h // 2)

        if angle > 0:
            rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
            mask = cv2.warpAffine(mask, rot_mat, (mask_w, mask_h), flags=cv2.INTER_LINEAR)

        offset = (random.randint(-16, 16), random.randint(16, 16))
        center = (center[0] + offset[0], center[1] + offset[1])
        mask = mask[center[1] - h // 2: center[1] + h // 2, center[0] - w // 2: center[0] + w // 2, :]
        mask = cv2.resize(mask, (w, h))
        assert img.shape == mask.shape
        img = np.where(mask == 0, img, color).astype('uint8')
        return img


class RandomFigures(object):
    """Insert random figure or some figures from the list [line, rectangle, circle]
    with random color and thickness
    """

    def __init__(self, p=0.5, random_color=True, always_single_figure=False,
                 thicknesses=(1, 6), circle_radiuses=(5, 64), figure_prob=0.5,
                 figures=None, **kwargs):
        self.p = p
        self.random_color = random_color
        self.always_single_figure = always_single_figure
        self.thicknesses = thicknesses
        self.circle_radiuses = circle_radiuses
        self.figure_prob = figure_prob

        if figures is None:
            self.figures = [cv2.line, cv2.rectangle, cv2.circle]
        else:
            assert isinstance(figures, (tuple, list))

            self.figures = []
            for figure in figures:
                assert isinstance(figure, str)

                if hasattr(cv2, figure):
                    self.figures.append(getattr(cv2, figure))
                else:
                    raise ValueError('Unknown figure: {}'.format(figure))

    def __call__(self, input_tuple):
        if random.uniform(0, 1) > self.p:
            return input_tuple

        cv_image, mask = input_tuple

        if self.always_single_figure:
            figure = [self.figures[random.randint(0, len(self.figures) - 1)]]
        else:
            figure = []
            for i in range(len(self.figures)):
                if random.uniform(0, 1) > self.figure_prob:
                    figure.append(self.figures[i])

        h, w = cv_image.shape[:2]
        for f in figure:
            p1 = (random.randint(0, w), random.randint(0, h))
            p2 = (random.randint(0, w), random.randint(0, h))
            color = tuple([random.randint(0, 256) for _ in range(3)]) if self.random_color else (0, 0, 0)
            thickness = random.randint(*self.thicknesses)
            if f != cv2.circle:
                cv_image = f(cv_image, p1, p2, color, thickness)
            else:
                r = random.randint(*self.circle_radiuses)
                cv_image = f(cv_image, p1, r, color, thickness)

        return Image.fromarray(cv_image), mask


class CutOutWithPrior(object):
    """Cut out around facial landmarks
    """

    def __init__(self, max_area=0.1, p=0.5, **kwargs):
        self.p = p
        self.max_area = max_area

    def __call__(self, input_tuple, *args, **kwargs):
        img, mask = input_tuple

        img = np.array(img)

        height, width = img.shape[:2]
        keypoints_ref = np.zeros((5, 2), dtype=np.float32)
        keypoints_ref[:, 0] = FivePointsAligner.ref_landmarks[:, 0] * width
        keypoints_ref[:, 1] = FivePointsAligner.ref_landmarks[:, 1] * height

        if float(torch.FloatTensor(1).uniform_()) < self.p:
            erase_num = torch.LongTensor(1).random_(1, 4)
            erase_ratio = torch.FloatTensor(1).uniform_(self.max_area / 2, self.max_area)
            erase_h = math.sqrt(erase_ratio) / float(erase_num) * height
            erase_w = math.sqrt(erase_ratio) / float(erase_num) * width

            erased_idx = []
            for _ in range(erase_num):
                erase_pos = int(torch.LongTensor(1).random_(0, 5))
                while erase_pos in erased_idx:
                    erase_pos = int(torch.LongTensor(1).random_(0, 5))

                left_corner = (
                    int(keypoints_ref[erase_pos][0] - erase_h / 2), int(keypoints_ref[erase_pos][1] - erase_w / 2))
                right_corner = (
                    int(keypoints_ref[erase_pos][0] + erase_h / 2), int(keypoints_ref[erase_pos][1] + erase_w / 2))

                cv2.rectangle(img, tuple(left_corner), tuple(right_corner), (0, 0, 0), thickness=-1)
                erased_idx.append(erase_pos)

        img = Image.fromarray(img)
        return img, mask


class GaussianBlur(object):
    """Apply gaussian blur with random parameters
    """

    def __init__(self, p, k):
        self.p = p
        assert k % 2 == 1
        self.k = k

    def __call__(self, input_tuple, *args, **kwargs):
        if random.uniform(0, 1) > self.p:
            return input_tuple

        img, mask = input_tuple

        img = np.array(img)
        img = cv2.blur(img, (self.k, self.k))
        img = Image.fromarray(img)

        return img, mask


class GaussianNoise(object):
    """Adds gaussian noise with random parameters
    """

    def __init__(self, p, sigma, grayscale):
        self.p = p
        self.sigma = sigma
        self.grayscale = grayscale

    def __call__(self, input_tuple, *args, **kwargs):
        if random.uniform(0, 1) > self.p:
            return input_tuple

        img, mask = input_tuple

        image_max_brightness = np.max(img)
        image_min_brightness = np.min(img)
        brightness_range = image_max_brightness - image_min_brightness
        max_noise_sigma = self.sigma * float(brightness_range if brightness_range > 0 else 1)
        noise_sigma = np.random.uniform(0, max_noise_sigma)

        img = np.array(img, dtype=np.float32)

        noise_shape = img.shape[:2] + (1,) if self.grayscale else img.shape
        img += np.random.normal(loc=0.0, scale=noise_sigma, size=noise_shape)

        img[img < 0.0] = 0.0
        img[img > 255.0] = 255.0

        img = Image.fromarray(img.astype(np.uint8))

        return img, mask


class MixUp(object):
    """MixUp augmentation
    """

    def __init__(self, p=0.5, alpha=0.2, images_root_dir=None, images_list_file=None, **kwargs):
        self.p = p
        self.alpha = alpha

        self.enable = images_root_dir is not None and exists(images_root_dir) and \
                      images_list_file is not None and exists(images_list_file)

        if self.enable:
            all_image_files = []
            with open(images_list_file) as input_stream:
                for line in input_stream:
                    image_name = line.replace('\n', '')
                    if len(image_name) > 0:
                        image_path = join(images_root_dir, image_name)
                        all_image_files.append(image_path)
            self.surrogate_image_paths = all_image_files
            assert len(self.surrogate_image_paths) > 0

    def get_num_images(self):
        return len(self.surrogate_image_paths) if self.enable else 0

    def _load_surrogate_image(self, idx, trg_image_size):
        trg_image_width, trg_image_height = trg_image_size

        image = read_image(self.surrogate_image_paths[idx])

        scale = 1.125
        new_width, new_height = int(trg_image_width * scale), int(trg_image_height * scale)
        image = image.resize((new_width, new_height), Image.BILINEAR)

        x_max_range = new_width - trg_image_width
        y_max_range = new_height - trg_image_height
        x1 = int(round(random.uniform(0, x_max_range)))
        y1 = int(round(random.uniform(0, y_max_range)))

        image = image.crop((x1, y1, x1 + trg_image_width, y1 + trg_image_height))

        if random.uniform(0, 1) > 0.5:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)

        return image

    def __call__(self, input_tuple, *args, **kwargs):
        if not self.enable or random.uniform(0, 1) > self.p:
            return input_tuple

        img, mask = input_tuple

        alpha = np.random.beta(self.alpha, self.alpha)
        alpha = alpha if alpha < 0.5 else 1.0 - alpha

        surrogate_image_idx = random.randint(0, len(self.surrogate_image_paths) - 1)
        surrogate_image = self._load_surrogate_image(surrogate_image_idx, img.size)

        float_img = np.array(img).astype(np.float32)
        float_surrogate_image = np.array(surrogate_image).astype(np.float32)
        mixed_image = (1.0 - alpha) * float_img + alpha * float_surrogate_image

        out_image = mixed_image.clip(0.0, 255.0).astype(np.uint8)

        return Image.fromarray(out_image), mask


class RandomBackgroundSubstitution(object):
    """Random background substitution augmentation
    """

    def __init__(self, p=0.5, images_root_dir=None, images_list_file=None, **kwargs):
        self.p = p

        self.enable = images_root_dir is not None and exists(images_root_dir) and \
                      images_list_file is not None and exists(images_list_file)

        if self.enable:
            all_image_files = []
            with open(images_list_file) as input_stream:
                for line in input_stream:
                    image_name = line.replace('\n', '')
                    if len(image_name) > 0:
                        image_path = join(images_root_dir, image_name)
                        all_image_files.append(image_path)
            self.surrogate_image_paths = all_image_files
            assert len(self.surrogate_image_paths) > 0

    def get_num_images(self):
        return len(self.surrogate_image_paths) if self.enable else 0

    def _load_bg_image(self, idx, trg_image_size):
        trg_image_width, trg_image_height = trg_image_size

        image = read_image(self.surrogate_image_paths[idx])

        scale = 1.125
        new_width, new_height = int(trg_image_width * scale), int(trg_image_height * scale)
        image = image.resize((new_width, new_height), Image.BILINEAR)

        x_max_range = new_width - trg_image_width
        y_max_range = new_height - trg_image_height
        x1 = int(round(random.uniform(0, x_max_range)))
        y1 = int(round(random.uniform(0, y_max_range)))

        image = image.crop((x1, y1, x1 + trg_image_width, y1 + trg_image_height))

        if random.uniform(0, 1) > 0.5:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)

        return image

    def __call__(self, input_tuple, *args, **kwargs):
        if not self.enable or random.uniform(0, 1) > self.p:
            return input_tuple

        img, mask = input_tuple
        if mask == '':
            return input_tuple

        surrogate_image_idx = random.randint(0, len(self.surrogate_image_paths) - 1)
        surrogate_image = self._load_bg_image(surrogate_image_idx, img.size)

        src_img = np.array(img)
        out_img = np.array(surrogate_image)

        fg_mask = np.array(mask).astype(np.bool)
        out_img[fg_mask] = src_img[fg_mask]

        return Image.fromarray(out_img), mask


class DisableBackground(object):
    def __call__(self, input_tuple, *args, **kwargs):
        img, mask = input_tuple
        if mask == '':
            return input_tuple

        bg_mask = ~mask.astype(np.bool)

        img = img
        img[bg_mask] = 0

        return Image.fromarray(img), mask

class RandomCropPad(TorchRandomCrop):
    def __init__(self, size, padding):
        super().__init__(size=size, padding=padding)

    def __call__(self, input_tuple):
        image, mask = input_tuple
        image = self.forward(image)
        mask = self.forward(mask) if mask else ''

        return image, mask


def ocv_resize_2_pil(img, size, interp=cv2.INTER_LINEAR, to_pill=True):
    resized = cv2.resize(img, dsize=size, interpolation=interp)
    if to_pill:
        return Image.fromarray(resized)
    return resized


class PairResize:
    def __init__(self, size, interpolation=cv2.INTER_LINEAR, to_pill=True):
        assert isinstance(size, int) or len(size) == 2
        self.size = size
        self.interpolation = interpolation
        self.to_pill = to_pill

    def __call__(self, input_tuple):
        image, mask = input_tuple

        image = ocv_resize_2_pil(image, self.size, self.interpolation, self.to_pill)
        mask = ocv_resize_2_pil(mask, self.size, self.interpolation, self.to_pill) if mask != '' else mask

        return image, mask


class SingleResize:
    def __init__(self, size, interpolation=cv2.INTER_LINEAR):
        assert isinstance(size, int) or len(size) == 2
        self.size = size
        self.interpolation = interpolation

    def __call__(self, image):
        image = ocv_resize_2_pil(image, self.size, self.interpolation)
        return image


class ToPILL:
    def __call__(self, input_tuple):
        image, mask = input_tuple
        mask = Image.fromarray(mask) if mask != '' else mask
        image = Image.fromarray(image)
        return image, mask


class PairNormalize(object):
    def __init__(self, mean, std, inplace=False):
        self.mean = mean
        self.std = std
        self.inplace = inplace

    def __call__(self, input_tuple):
        image, mask = input_tuple

        return F.normalize(image, self.mean, self.std, self.inplace), mask


class PairToTensor(object):
    def __call__(self, input_tuple):
        image, mask = input_tuple

        image = F.to_tensor(image)
        mask = F.to_tensor(mask) if mask != '' else mask

        return image, mask

_AUGMIX_TRANSFORMS_GREY = [
            'SharpnessIncreasing',  # not in paper
            'ShearX',
            'ShearY',
            'TranslateXRel',
            'TranslateYRel',
        ]

_AUGMIX_TRANSFORMS = [
            'AutoContrast',
            'ColorIncreasing',  # not in paper
            'ContrastIncreasing',  # not in paper
            'BrightnessIncreasing',  # not in paper
            'SharpnessIncreasing',  # not in paper
            'Equalize',
            'PosterizeIncreasing',
            'SolarizeIncreasing',
            'ShearX',
            'ShearY',
            'TranslateXRel',
            'TranslateYRel',
        ]


class OpsFabric:
    def __init__(self, name, magnitude, hparams, prob=1.0):
        self.max_level = 10
        self.prob = prob
        self.hparams = hparams
        # kwargs for augment functions
        self.aug_kwargs = dict(
            fillcolor=hparams['img_mean'],
            resample=(Image.BILINEAR, Image.BICUBIC)
        )
        self.LEVEL_TO_ARG = {
            'AutoContrast': None,
            'Equalize': None,
            'Rotate': self._rotate_level_to_arg,
            'PosterizeIncreasing': self._posterize_increasing_level_to_arg,
            'SolarizeIncreasing': self._solarize_increasing_level_to_arg,
            'ColorIncreasing': self._enhance_increasing_level_to_arg,
            'ContrastIncreasing': self._enhance_increasing_level_to_arg,
            'BrightnessIncreasing': self._enhance_increasing_level_to_arg,
            'SharpnessIncreasing': self._enhance_increasing_level_to_arg,
            'ShearX': self._shear_level_to_arg,
            'ShearY': self._shear_level_to_arg,
            'TranslateXRel': self._translate_rel_level_to_arg,
            'TranslateYRel': self._translate_rel_level_to_arg,
        }
        self.NAME_TO_OP = {
            'AutoContrast': self.auto_contrast,
            'Equalize': self.equalize,
            'Rotate': self.rotate,
            'PosterizeIncreasing': self.posterize,
            'SolarizeIncreasing': self.solarize,
            'ColorIncreasing': self.color,
            'ContrastIncreasing': self.contrast,
            'BrightnessIncreasing': self.brightness,
            'SharpnessIncreasing': self.sharpness,
            'ShearX': self.shear_x,
            'ShearY': self.shear_y,
            'TranslateXRel': self.translate_x_rel,
            'TranslateYRel': self.translate_y_rel,
        }
        self.aug_fn = self.NAME_TO_OP[name]
        self.level_fn = self.LEVEL_TO_ARG[name]
        self.magnitude = magnitude
        self.magnitude_std = self.hparams.get('magnitude_std', float('inf'))

    @staticmethod
    def check_args_tf(kwargs):
        def _interpolation(kwargs):
            interpolation = kwargs.pop('resample', Image.BILINEAR)
            if isinstance(interpolation, (list, tuple)):
                return random.choice(interpolation)
            else:
                return interpolation

        kwargs['resample'] = _interpolation(kwargs)

    @staticmethod
    def auto_contrast(img, **__):
        return ImageOps.autocontrast(img)

    @staticmethod
    def equalize(img, **__):
        return ImageOps.equalize(img)

    @staticmethod
    def solarize(img, thresh, **__):
        return ImageOps.solarize(img, thresh)

    @staticmethod
    def posterize(img, bits_to_keep, **__):
        if bits_to_keep >= 8:
            return img
        return ImageOps.posterize(img, bits_to_keep)

    @staticmethod
    def contrast(img, factor, **__):
        return ImageEnhance.Contrast(img).enhance(factor)

    @staticmethod
    def color(img, factor, **__):
        return ImageEnhance.Color(img).enhance(factor)

    @staticmethod
    def brightness(img, factor, **__):
        return ImageEnhance.Brightness(img).enhance(factor)

    @staticmethod
    def sharpness(img, factor, **__):
        return ImageEnhance.Sharpness(img).enhance(factor)

    @staticmethod
    def randomly_negate(v):
        """With 50% prob, negate the value"""
        return -v if random.random() > 0.5 else v

    def shear_x(self, img, factor, **kwargs):
        self.check_args_tf(kwargs)
        return img.transform(img.size, Image.AFFINE, (1, factor, 0, 0, 1, 0), **kwargs)

    def shear_y(self, img, factor, **kwargs):
        self.check_args_tf(kwargs)
        return img.transform(img.size, Image.AFFINE, (1, 0, 0, factor, 1, 0), **kwargs)

    def translate_x_rel(self, img, pct, **kwargs):
        pixels = pct * img.size[0]
        self.check_args_tf(kwargs)
        return img.transform(img.size, Image.AFFINE, (1, 0, pixels, 0, 1, 0), **kwargs)

    def translate_y_rel(self, img, pct, **kwargs):
        pixels = pct * img.size[1]
        self.check_args_tf(kwargs)
        return img.transform(img.size, Image.AFFINE, (1, 0, 0, 0, 1, pixels), **kwargs)

    def rotate(self, img, degrees, **kwargs):
        self.check_args_tf(kwargs)
        return img.rotate(degrees, **kwargs)

    def _rotate_level_to_arg(self, level, _hparams):
        # range [-30, 30]
        level = (level / self.max_level) * 30.
        level = self.randomly_negate(level)
        return level,

    def _enhance_increasing_level_to_arg(self, level, _hparams):
        # range [0.1, 1.9]
        level = (level / self.max_level) * .9
        level = 1.0 + self.randomly_negate(level)
        return level,

    def _shear_level_to_arg(self, level, _hparams):
        # range [-0.3, 0.3]
        level = (level / self.max_level) * 0.3
        level = self.randomly_negate(level)
        return level,

    def _translate_rel_level_to_arg(self, level, hparams):
        # default range [-0.45, 0.45]
        translate_pct = hparams.get('translate_pct', 0.45)
        level = (level / self.max_level) * translate_pct
        level = self.randomly_negate(level)
        return level,

    def _posterize_level_to_arg(self, level, _hparams):
        # range [0, 4], 'keep 0 up to 4 MSB of original image'
        # intensity/severity of augmentation decreases with level
        return int((level / self.max_level) * 4),

    def _posterize_increasing_level_to_arg(self, level, hparams):
        # range [4, 0], 'keep 4 down to 0 MSB of original image',
        # intensity/severity of augmentation increases with level
        return 4 - self._posterize_level_to_arg(level, hparams)[0],

    def _solarize_level_to_arg(self, level, _hparams):
        # range [0, 256]
        # intensity/severity of augmentation decreases with level
        return int((level / self.max_level) * 256),

    def _solarize_increasing_level_to_arg(self, level, _hparams):
        # range [0, 256]
        # intensity/severity of augmentation increases with level
        return 256 - self._solarize_level_to_arg(level, _hparams)[0],

    def __call__(self, img):
        if self.prob < 1.0 and random.random() > self.prob:
            return img
        magnitude = self.magnitude
        if self.magnitude_std:
            if self.magnitude_std == float('inf'):
                magnitude = random.uniform(0, magnitude)
            elif self.magnitude_std > 0:
                magnitude = random.gauss(magnitude, self.magnitude_std)
        magnitude = min(self.max_level, max(0, magnitude))  # clip to valid range
        level_args = self.level_fn(magnitude, self.hparams) if self.level_fn is not None else tuple()
        return self.aug_fn(img, *level_args, **self.aug_kwargs)


class AugMixAugment:
    """ AugMix Transform
    Adapted and improved from impl here: https://github.com/google-research/augmix/blob/master/imagenet.py
    From paper: 'AugMix: A Simple Data Processing Method to Improve Robustness and Uncertainty -
    https://arxiv.org/abs/1912.02781
    """
    def __init__(self, ops, alpha=1., width=3, depth=-1, **kwargs):
        self.ops = ops
        self.alpha = alpha
        self.width = width
        self.depth = depth

    def _apply_basic(self, img, mixing_weights, m):
        # This is a literal adaptation of the paper/official implementation without normalizations and
        # PIL <-> Numpy conversions between every op. It is still quite CPU compute heavy compared to the
        # typical augmentation transforms, could use a GPU / Kornia implementation.
        img_shape = img.size[0], img.size[1], len(img.getbands())
        mixed = np.zeros(img_shape, dtype=np.float32)
        for mw in mixing_weights:
            depth = self.depth if self.depth > 0 else np.random.randint(1, 4)
            ops = np.random.choice(self.ops, depth, replace=True)
            img_aug = img  # no ops are in-place, deep copy not necessary
            for op in ops:
                img_aug = op(img_aug)
            mixed += mw * np.asarray(img_aug, dtype=np.float32)
        np.clip(mixed, 0, 255., out=mixed)
        mixed = Image.fromarray(mixed.astype(np.uint8))
        return Image.blend(img, mixed, m)

    def __call__(self, input_tuple):
        img, mask = input_tuple
        mixing_weights = np.float32(np.random.dirichlet([self.alpha] * self.width))
        m = np.float32(np.random.beta(self.alpha, self.alpha))
        mixed = self._apply_basic(img, mixing_weights, m)
        mask = self._apply_basic(mask, mixing_weights, m) if mask else ''
        return mixed, mask

def augment_and_mix_transform(config_str, image_mean, translate_const=250, grey=False):
    """ Create AugMix PyTorch transform
    :param config_str: String defining configuration of random augmentation. Consists of multiple sections separated by
    dashes ('-'). The first section defines the specific variant of rand augment (currently only 'rand'). The remaining
    sections, not order sepecific determine
        'm' - integer magnitude (severity) of augmentation mix (default: 3)
        'w' - integer width of augmentation chain (default: 3)
        'd' - integer depth of augmentation chain (-1 is random [1, 3], default: -1)
        'b' - integer (bool), blend each branch of chain into end result without a final blend, less CPU (default: 0)
        'mstd' -  float std deviation of magnitude noise applied (default: 0)
    Ex 'augmix-m5-w4-d2' results in AugMix with severity 5, chain width 4, chain depth 2
    :param hparams: Other hparams (kwargs) for the Augmentation transforms
    :return: A PyTorch compatible Transform
    imported and modified from: https://github.com/rwightman/pytorch-image-models/blob/master/timm/data/auto_augment.py
    """
    def augmix_ops(magnitude, hparams, prob=1.0, grey=False):
        aug_politics = _AUGMIX_TRANSFORMS_GREY if grey else _AUGMIX_TRANSFORMS
        return [OpsFabric(name, magnitude, hparams, prob) for name in aug_politics]

    magnitude = 3
    width = 3
    depth = -1
    alpha = 1.
    p=1.0
    hparams = dict(
            translate_const=translate_const,
            img_mean=tuple([int(c * 256) for c in image_mean]),
            magnitude_std=float('inf')
        )
    config = config_str.split('-')
    assert config[0] == 'augmix'
    config = config[1:]
    for c in config:
        cs = re.split(r'(\d.*)', c)
        if len(cs) < 2:
            continue
        key, val = cs[:2]
        if key == 'mstd':
            hparams.setdefault('magnitude_std', float(val))
        elif key == 'm':
            magnitude = int(val)
        elif key == 'w':
            width = int(val)
        elif key == 'd':
            depth = int(val)
        elif key == 'a':
            alpha = float(val)
        elif key == 'p':
            p = float(val)
        else:
            assert False, 'Unknown AugMix config section'
    ops = augmix_ops(magnitude=magnitude, hparams=hparams, prob=p, grey=grey)
    return AugMixAugment(ops, alpha=alpha, width=width, depth=depth)


def build_transforms(height, width, transforms=None, norm_mean=(0.485, 0.456, 0.406),
                     norm_std=(0.229, 0.224, 0.225), apply_masks_to_test=False, **kwargs):
    """Builds train and test transform functions.

    Args:
        height (int): target image height.
        width (int): target image width.
        transforms (str or list of str, optional): transformations applied to model training.
            Default is 'random_flip'.
        norm_mean (list or None, optional): normalization mean values. Default is ImageNet means.
        norm_std (list or None, optional): normalization standard deviation values. Default is
            ImageNet standard deviation values.
    """
    if transforms is None:
        return None, None

    if norm_mean is None or norm_std is None:
        norm_mean = [0.485, 0.456, 0.406]  # imagenet mean
        norm_std = [0.229, 0.224, 0.225]  # imagenet std

    print('Building train transforms ...')
    transform_tr = []
    if transforms.random_grid.enable:
        print('+ random_grid')
        transform_tr += [RandomGrid(**transforms.random_grid)]
    if transforms.random_figures.enable:
        print('+ random_figures')
        transform_tr += [RandomFigures(**transforms.random_figures)]
    if transforms.center_crop.enable and not transforms.center_crop.test_only:
        print('+ center_crop')
        transform_tr += [CenterCrop(margin=transforms.center_crop.margin)]
    if transforms.random_crop.enable:
        print('+ random crop')
        transform_tr += [RandomCrop(p=transforms.random_crop.p,
                                    scale=transforms.random_crop.scale,
                                    margin=transforms.random_crop.margin,
                                    align_ar=transforms.random_crop.scale,
                                    align_center=transforms.random_crop.align_center,
                                    target_ar=float(height)/float(width))]
    print('+ resize to {}x{}'.format(height, width))
    transform_tr += [PairResize((height, width))]
    if transforms.augmix.enable:
        print('+ AugMix')
        transform_tr += [augment_and_mix_transform(config_str=transforms.augmix.cfg_str, image_mean=norm_mean, grey=transforms.augmix.grey_imgs)]
    if transforms.randaugment.enable:
        print('+ RandAugment')
        transform_tr += [RandomAugment()]
    if transforms.cutout.enable:
        print('+ Cutout')
        transform_tr += [Cutout(**transforms.cutout)]
    if transforms.random_background_substitution.enable:
        aug_module = RandomBackgroundSubstitution(**transforms.random_background_substitution)
        if aug_module.enable:
            print('+ random_background_substitution')
            transform_tr += [aug_module]
    if transforms.random_flip.enable:
        print('+ random flip')
        transform_tr += [RandomHorizontalFlip(p=transforms.random_flip.p)]
    if transforms.cut_out_with_prior.enable:
        print('+ cut out with prior')
        transform_tr += [CutOutWithPrior(p=transforms.cut_out_with_prior.p,
                                         max_area=transforms.cut_out_with_prior.max_area)]
    if transforms.random_blur.enable:
        print('+ random_blur')
        transform_tr += [GaussianBlur(p=transforms.random_blur.p,
                                      k=transforms.random_blur.k)]
    if transforms.random_noise.enable:
        print('+ random_noise')
        transform_tr += [GaussianNoise(p=transforms.random_noise.p,
                                       sigma=transforms.random_noise.sigma,
                                       grayscale=transforms.random_noise.grayscale)]
    if transforms.mixup.enable:
        mixup_augmentor = MixUp(**transforms.mixup)
        print('+ mixup (with {} extra images)'.format(mixup_augmentor.get_num_images()))
        transform_tr += [mixup_augmentor]
    if transforms.random_patch.enable:
        print('+ random patch')
        transform_tr += [RandomPatch(**transforms.random_patch)]
    if transforms.color_jitter.enable:
        print('+ color jitter')
        transform_tr += [RandomColorJitter(**transforms.color_jitter)]
    if transforms.random_gray_scale.enable:
        print('+ random_gray_scale')
        transform_tr += [RandomGrayscale(p=transforms.random_gray_scale.p)]
    if transforms.random_rotate.enable:
        print('+ random_rotate')
        transform_tr += [RandomRotate(**transforms.random_rotate)]
    if transforms.equalize.enable:
        print('+ equalize')
        transform_tr += [Equalize(**transforms.equalize)]
    if transforms.posterize.enable:
        print('+ posterize')
        transform_tr += [Posterize(**transforms.posterize)]
    if transforms.random_erase.enable and not transforms.random_erase.norm_image:
        print('+ random erase')
        transform_tr += [RandomErasing(**transforms.random_erase)]
    if transforms.random_negative.enable:
        print('+ random negative')
        transform_tr += [RandomNegative(**transforms.random_negative)]
    if transforms.force_gray_scale.enable:
        print('+ force_gray_scale')
        transform_tr += [ForceGrayscale()]
    if transforms.coarse_dropout.enable:
        print('+ coarse_dropout')
        transform_tr += [CoarseDropout(**transforms.coarse_dropout)]
    if transforms.crop_pad.enable:
        print('+ crop_pad')
        transform_tr += [RandomCropPad((height, width), padding=int(0.125*height))]

    print('+ to torch tensor of range [0, 1]')
    transform_tr += [PairToTensor()]
    print('+ normalization (mean={}, std={})'.format(norm_mean, norm_std))
    transform_tr += [PairNormalize(mean=norm_mean, std=norm_std)]
    if transforms.random_erase.enable and transforms.random_erase.norm_image:
        print('+ random erase')
        transform_tr += [RandomErasing(**transforms.random_erase)]

    transform_tr = Compose(transform_tr)
    transform_te = build_test_transform(height, width, norm_mean, norm_std, apply_masks_to_test, transforms)
    return transform_tr, transform_te


def build_test_transform(height, width, norm_mean=(0.485, 0.456, 0.406), norm_std=(0.229, 0.224, 0.225),
                         apply_masks_to_test=False, transforms=None, **kwargs):
    def get_resize(h, w, scale, to_pill=True):
        t_h, t_w = int(h * scale), int(w * scale)
        print('+ resize to {}x{}'.format(t_h, t_w))
        return PairResize((t_h, t_w), to_pill=to_pill)
    print('Building test transforms ...')
    transform_te = []
    if transforms.test.resize_first:
        transform_te.append(get_resize(height, width, transforms.test.resize_scale, to_pill=False))
    if transforms.center_crop.enable:
        print('+ center_crop')
        transform_te.append(CenterCrop(margin=transforms.center_crop.margin))
    if apply_masks_to_test:
        print('+ background zeroing')
        transform_te.append(DisableBackground())
    if not transforms.test.resize_first:
        transform_te.append(get_resize(height, width, transforms.test.resize_scale))
    else:
        transform_te.append(ToPILL())
    if transforms is not None and transforms.force_gray_scale.enable:
        print('+ force_gray_scale')
        transform_te.append(ForceGrayscale())
    print('+ to torch tensor of range [0, 1]')
    transform_te.append(PairToTensor())
    print('+ normalization (mean={}, std={})'.format(norm_mean, norm_std))
    transform_te.append(PairNormalize(mean=norm_mean, std=norm_std))
    transform_te = Compose(transform_te)

    return transform_te


def build_inference_transform(height, width, norm_mean=(0.485, 0.456, 0.406),
                              norm_std=(0.229, 0.224, 0.225), **kwargs):
    print('Building inference transforms ...')
    print('+ resize to {}x{}'.format(height, width))
    print('+ to torch tensor of range [0, 1]')
    print('+ normalization (mean={}, std={})'.format(norm_mean, norm_std))
    transform_te = Compose([
        SingleResize((height, width)),
        ToTensor(),
        Normalize(mean=norm_mean, std=norm_std),
    ])

    return transform_te
