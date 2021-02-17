# Source: https://github.com/pytorch/vision/blob/master/references/detection/transforms.py
# Also: https://pytorch.org/docs/stable/_modules/torchvision/transforms/transforms.html

import random
import torch
import numbers
from torchvision.transforms import functional as F
from copy import deepcopy
import numpy as np
from PIL import Image
from histomicstk.preprocessing.augmentation import \
    rgb_perturb_stain_concentration
from torchvision.ops import boxes as box_ops
from collections.abc import Sequence


_pil_interpolation_to_str = {
    Image.NEAREST: 'PIL.Image.NEAREST',
    Image.BILINEAR: 'PIL.Image.BILINEAR',
    Image.BICUBIC: 'PIL.Image.BICUBIC',
    Image.LANCZOS: 'PIL.Image.LANCZOS',
    Image.HAMMING: 'PIL.Image.HAMMING',
    Image.BOX: 'PIL.Image.BOX',
}


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


def _get_image_size(img):
    if F._is_pil_image(img):
        return img.size
    elif isinstance(img, torch.Tensor) and img.dim() > 2:
        return img.shape[-2:][::-1]
    else:
        raise TypeError("Unexpected type {}".format(type(img)))


def remove_degenerate_bboxes(boxes, dim0: int, dim1: int, min_boxside=0):
    """Remove bboxes beyond image or smaller than a minimum size.
    This assumes a format where columns 0, 1 are min dim0 and dim1, while
    columns 2, 3 are max dim0 and dim1, respectively.
    """
    # adjust boxes
    boxes[boxes < 0] = 0
    boxes[boxes[:, 0] > dim0, 0] = dim0
    boxes[boxes[:, 2] > dim0, 2] = dim0
    boxes[boxes[:, 1] > dim1, 1] = dim1
    boxes[boxes[:, 3] > dim1, 3] = dim1

    # remove boxes outside cropped region
    keep1 = boxes[:, 3] - boxes[:, 1] > min_boxside
    keep2 = boxes[:, 2] - boxes[:, 0] > min_boxside
    keep = keep1 & keep2
    boxes = boxes[keep]

    return boxes, keep


# noinspection LongLine
class Cropper(object):
    """Crop the given PIL Image at a specific/random location.

    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (dim1, dim0), a square crop (size, size) is
            made.
        padding (int or sequence, optional): Optional padding on each border
            of the image. Default is None, i.e no padding. If a sequence of length
            4 is provided, it is used to pad left, top, right, bottom borders
            respectively. If a sequence of length 2 is provided, it is used to
            pad left/right, top/bottom borders, respectively.
        pad_if_needed (boolean): It will pad the image if smaller than the
            desired size to avoid raising an exception. Since cropping is done
            after padding, the padding seems to be done at a random offset.
        fill: Pixel fill value for constant fill. Default is 0. If a tuple of
            length 3, it is used to fill R, G, B channels respectively.
            This value is only used when the padding_mode is constant
        padding_mode: Type of padding. Should be: constant, edge, reflect or symmetric. Default is constant.

             - constant: pads with a constant value, this value is specified with fill

             - edge: pads with the last value on the edge of the image

             - reflect: pads with reflection of image (without repeating the last value on the edge)

                padding [1, 2, 3, 4] with 2 elements on both sides in reflect mode
                will result in [3, 2, 1, 2, 3, 4, 3, 2]

             - symmetric: pads with reflection of image (repeating the last value on the edge)

                padding [1, 2, 3, 4] with 2 elements on both sides in symmetric mode
                will result in [2, 1, 1, 2, 3, 4, 4, 3]

    """

    def __init__(
            self, size=None, plusminus=None, padding=None, pad_if_needed=True,
            fill=0, padding_mode='constant', min_boxside=8, iscentral=False):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))  # width, height
        else:
            self.size = size
        self.plusminus = plusminus
        self.padding = padding
        self.pad_if_needed = pad_if_needed
        self.fill = fill
        self.padding_mode = padding_mode
        self.min_boxside = min_boxside
        self.iscentral = iscentral  # central crop? (otherwise random)

        if plusminus is not None:
            assert plusminus < self.size[0]

    def get_params(self, img, output_size):
        """Get parameters for ``crop`` for a random crop.

        Args:
            img (PIL Image): Image to be cropped.
            output_size (tuple): Expected cropped output size (width, height)

        Returns:
            tuple: params (i, j, dim1, dim0) to be passed to ``crop`` for random crop.
        """
        w, h = _get_image_size(img)
        tw, th = output_size
        if w == tw and h == th:
            return 0, 0, h, w

        if self.iscentral:
            i = (h - th) // 2
            j = (w - tw) // 2
        else:
            i = random.randint(0, h - th)
            j = random.randint(0, w - tw)
        return i, j, th, tw

    def _crop_ops(self, im, size, i, j, h, w, get_params=True):
        # maybe pad
        if self.padding is not None:
            im = F.pad(im, self.padding, self.fill, self.padding_mode)

        # pad the width if needed
        wpad = max(0, size[0] - im.size[0])
        if self.pad_if_needed and wpad > 0:
            im = F.pad(im, (wpad, 0, 0, 0), self.fill, self.padding_mode)
        # pad the height if needed
        hpad = max(0, size[1] - im.size[1])
        if self.pad_if_needed and hpad > 0:
            im = F.pad(im, (0, hpad, 0, 0), self.fill, self.padding_mode)

        if get_params:
            i, j, h, w = self.get_params(im, size)
        im = F.crop(im, i, j, h, w)

        return im, wpad, hpad, i, j, h, w

    def __call__(self, rgb, targets=None, i=None, j=None, h=None, w=None):
        """
        Args:
            rgb (PIL Image): Image to be cropped.

        Returns:
            PIL Image: Cropped image.
        """
        # mFIXME?: do we really need a deep copy??
        img = deepcopy(rgb)
        target = deepcopy(targets)

        # determine crop region bounds
        if any([p is not None for p in [i, j, h, w]]):
            assert all([p is not None for p in [i, j, h, w]])
            size = (w, h)
            get_crop_params = False
        else:
            size = list(self.size)
            get_crop_params = True

            # Some randomization in crop size (it gets resized to
            #  a fixed size inside original model implementation in training
            #  mode) for scale augmentation (see Resnet paper, section 3.4):
            #  https://arxiv.org/pdf/1512.03385.pdf
            if self.plusminus is not None:
                w, h = _get_image_size(img)
                hpm = self.plusminus // 2
                pm = random.randint(-hpm, hpm)
                size[0] = min(size[0] + pm, w)
                size[1] = min(size[1] + pm, h)

        # crop rgb
        img, wpad, hpad, i, j, h, w = self._crop_ops(
            im=img, size=size, i=i, j=j, h=h, w=w, get_params=get_crop_params)

        # if we just want to crop the image
        if target is None:
            return img

        # crop dense mask. This is a histomicstk-style object mask
        # where the first channel encodes label, while the product of
        # the second and third channels encodes object id.
        # See: https://github.com/DigitalSlideArchive/HistomicsTK/ ...
        # .. blob/master/histomicstk/annotations_and_masks/ ..
        # .. annotations_to_object_mask_handler.py
        if 'dense_mask' in target:
            target['dense_mask'], _, _, _, _, _, _ = self._crop_ops(
                im=target['dense_mask'], size=size,
                i=i, j=j, h=h, w=w, get_params=False)

        # crop sparse mask
        if 'masks' in target:
            raise NotImplementedError(
                "Cropping sparse masks is not supported for now. It's a pain "
                "to implement since I have to deal with padding using tensor"
                "ops, and it's only relevant for the coco evaluator during"
                "testing .. otherwise I do all the cropping using the dense"
                "mask (eg for the data loader)."
            )

        if 'boxes' in target:
            # adjust boxes
            bbox = target['boxes']
            bbox[:, [0, 2]] += (wpad - j)
            bbox[:, [1, 3]] += (hpad - i)
            bbox, keep = remove_degenerate_bboxes(
                boxes=bbox.to(torch.float32), dim0=w, dim1=h,
                min_boxside=self.min_boxside)
            target['boxes'] = bbox

            if 'n_objects' in target:
                target['n_objects'] = torch.tensor([bbox.shape[0]])

            if 'area' in target:
                target['area'] = (
                    bbox[:, 3] - bbox[:, 1]) * (
                    bbox[:, 2] - bbox[:, 0])

            # crop vectors (classifications, tags etc)
            for key in ['labels', 'iscrowd', 'ismask', 'scores']:
                if key in target:
                    target[key] = target[key][keep]

        return img, target

    def __repr__(self):
        return self.__class__.__name__ + '(size={0}, padding={1})'.format(
            self.size, self.padding)


class RandomHorizontalFlip(object):
    def __init__(self, prob):
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob:
            width, height = _get_image_size(image)
            if F._is_pil_image(image):
                image = F.hflip(image)
            else:
                image = image.flip(-1)

            if "boxes" in target:
                bbox = target["boxes"]
                bbox[:, [0, 2]] = width - bbox[:, [2, 0]]
                target["boxes"] = bbox

            if 'dense_mask' in target:
                if F._is_pil_image(target['dense_mask']):
                    target['dense_mask'] = F.hflip(target['dense_mask'])
                else:
                    target['dense_mask'] = target['dense_mask'].flip(-1)

            if 'masks' in target:
                raise NotImplementedError(
                    "Flipping sparse masks is not supported for now.")

        return image, target


# noinspection LongLine
class ToPILImage(object):
    """Convert a tensor or an ndarray to PIL Image.

    Converts a torch.*Tensor of shape C x H x W or a numpy ndarray of shape
    H x W x C to a PIL Image while preserving the value range.

    Args:
        mode (`PIL.Image mode`_): color space and pixel depth of input data (optional).
            If ``mode`` is ``None`` (default) there are some assumptions made about the input data:
             - If the input has 4 channels, the ``mode`` is assumed to be ``RGBA``.
             - If the input has 3 channels, the ``mode`` is assumed to be ``RGB``.
             - If the input has 2 channels, the ``mode`` is assumed to be ``LA``.
             - If the input has 1 channel, the ``mode`` is determined by the data type (i.e ``int``, ``float``,
               ``short``).

    .. _PIL.Image mode: https://pillow.readthedocs.io/en/latest/handbook/concepts.html#concept-modes
    """
    def __init__(self, mode=None):
        self.mode = mode

    def __call__(self, pic, target):
        """
        Args:
            pic (Tensor or numpy.ndarray): Image to be converted to PIL Image.

        Returns:
            PIL Image: Image converted to PIL Image.

        """
        return F.to_pil_image(pic, self.mode), target

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        if self.mode is not None:
            format_string += 'mode={0}'.format(self.mode)
        format_string += ')'
        return format_string


# class ToTensor(object):
#     def __call__(self, image, target):
#         image = F.to_tensor(image)
#         return image, target


class PILToTensor(object):
    """Convert a ``PIL Image`` to a tensor of the same type.

    Converts a PIL Image (H x W x C) to a torch.Tensor of shape (C x H x W).
    """
    def __init__(self, float16=False):
        self.dtype = torch.float16 if float16 else torch.float32

    def __call__(self, pic, target, isuint8=True):
        """
        Args:
            pic (PIL Image): Image to be converted to tensor.

        Returns:
            Tensor: Converted image.
        """
        if 'mask' in target:
            raise NotImplementedError()

        # handle PIL Image
        img = torch.as_tensor(np.array(pic))
        img = img.view(pic.size[1], pic.size[0], len(pic.getbands()))
        # put it from HWC to CHW format
        img = img.permute((2, 0, 1))
        img = img.to(dtype=self.dtype)
        if isuint8:
            img = img / 255.
        return img, target

    def __repr__(self):
        return self.__class__.__name__ + '()'


class RandomHEStain(object):
    """Randomly perturn H&E concentrations. See docs for the function
    rgb_perturb_stain_concentration in Histomicstk module.
    """
    def __init__(self, sigma1=0.5, sigma2=0.5):
        self.sigma1 = sigma1
        self.sigma2 = sigma2

    def __call__(self, image, target):

        rgb = rgb_perturb_stain_concentration(
            im_rgb=np.array(image), sigma1=self.sigma1, sigma2=self.sigma2)
        return Image.fromarray(rgb), target


class RpnProposalAugmenter(object):
    """
    A test-time augmentation class that takes the output proposals from
    a region proposal network and randomly jitters them (eg shit, resize, etc)
    to obtain a realization that is slighty different .. therefore mapping to
    a different part of the feature map and resulting in slighly different
    box features whent he ROIAlign op is done.
    """
    def __init__(
            self, ops=None, max_shift=48,
            min_resize_factor=0.75, max_resize_factor=1.5):
        self.opmap = {
            'shift': self.shift,
            'resize': self.resize_with_same_aspect,
            'aspect': self.resize_with_random_aspect,
        }
        ops = ['shift'] if ops is None else ops
        self.ops = [self.opmap[op] for op in ops]
        self.max_shift = int(max_shift)
        self.min_resize_factor = min_resize_factor
        self.max_resize_factor = max_resize_factor

    def shift(self, boxes):
        shifts = torch.randint(
            -self.max_shift, self.max_shift,
            (boxes.shape[0], 2), device=boxes.device)
        shifts = torch.cat([shifts, shifts], 1)
        return boxes + shifts

    def _get_random_sfs(self, n, device):
        return 0.01 * torch.randint(
            int(100 * self.min_resize_factor),
            int(100 * self.max_resize_factor),
            (n,), device=device)

    # noinspection DuplicatedCode
    def _resize(self, boxes, same_aspect: bool):
        w_half = (boxes[:, 2] - boxes[:, 0]) * .5
        h_half = (boxes[:, 3] - boxes[:, 1]) * .5
        x_c = (boxes[:, 2] + boxes[:, 0]) * .5
        y_c = (boxes[:, 3] + boxes[:, 1]) * .5

        sf_x = self._get_random_sfs(boxes.shape[0], device=boxes.device)
        sf_y = sf_x if same_aspect else \
            self._get_random_sfs(boxes.shape[0], device=boxes.device)

        w_half *= sf_x
        h_half *= sf_y

        boxes_exp = torch.zeros_like(boxes, device=boxes.device)
        boxes_exp[:, 0] = x_c - w_half
        boxes_exp[:, 2] = x_c + w_half
        boxes_exp[:, 1] = y_c - h_half
        boxes_exp[:, 3] = y_c + h_half

        return boxes_exp

    def resize_with_same_aspect(self, boxes):
        return self._resize(boxes, same_aspect=True)

    def resize_with_random_aspect(self, boxes):
        return self._resize(boxes, same_aspect=False)

    def __call__(self, proposals, image_shapes):

        # randomly choose an augmentation op
        augmentop = random.choice(self.ops)

        prealization = []

        for pboxes, image_shape in zip(proposals, image_shapes):

            # noinspection PyArgumentList
            boxes = augmentop(pboxes)

            # make sure it still fits within the image bounds
            boxes = box_ops.clip_boxes_to_image(boxes, image_shape)

            prealization.append(boxes)

        return prealization


class Resize(torch.nn.Module):
    """Resize the input image to the given size.
    The image can be a PIL Image or a torch Tensor, in which case it is expected
    to have [..., H, W] shape, where ... means an arbitrary number of leading dimensions

    Args:
        size (sequence or int): Desired output size. If size is a sequence like
            (h, w), output size will be matched to this. If size is an int,
            smaller edge of the image will be matched to this number.
            i.e, if height > width, then image will be rescaled to
            (size * height / width, size).
            In torchscript mode padding as single int is not supported, use a tuple or
            list of length 1: ``[size, ]``.
        interpolation (int, optional): Desired interpolation enum defined by `filters`_.
            Default is ``PIL.Image.BILINEAR``. If input is Tensor, only ``PIL.Image.NEAREST``, ``PIL.Image.BILINEAR``
            and ``PIL.Image.BICUBIC`` are supported.
    """

    def __init__(self, size, interpolation=Image.BILINEAR):
        super().__init__()
        if not isinstance(size, (int, Sequence)):
            raise TypeError("Size should be int or sequence. Got {}".format(type(size)))
        if isinstance(size, Sequence) and len(size) not in (1, 2):
            raise ValueError("If size is a sequence, it should have 1 or 2 values")
        self.size = size
        self.interpolation = interpolation

    def forward(self, img):
        """
        Args:
            img (PIL Image or Tensor): Image to be scaled.

        Returns:
            PIL Image or Tensor: Rescaled image.
        """
        return F.resize(img, self.size, self.interpolation)


    def __repr__(self):
        interpolate_str = _pil_interpolation_to_str[self.interpolation]
        return self.__class__.__name__ + '(size={0}, interpolation={1})'.format(self.size, interpolate_str)

