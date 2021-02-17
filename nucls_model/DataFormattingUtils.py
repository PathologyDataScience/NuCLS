# Modified from Matthias Hamacher
# https://github.com/CancerDataScience/CellDetectionCenterpoint/

import torch
import torchvision
import numpy as np
from pandas import DataFrame
from itertools import combinations
from histomicstk.annotations_and_masks.masks_to_annotations_handler import \
    get_contours_from_mask
from configs.nucleus_style_defaults import DefaultAnnotationStyles


def image_to_tensor(image, cfg):
    """
    transforms and image to a tensor
    Args:
        image: image array of size (x, y, 3)
        cfg: configuration object or dict
    Returns: tensor of size (1, 3, x, y)
    """
    device = cfg['device']
    image = torch.tensor([image[..., 0], image[..., 1], image[..., 2]],
                         device=device).float()
    if cfg['im_normalize']:  # (0,0,0)(0.2,0.4,1)
        image = torchvision.transforms.Normalize(
            cfg['im_norm_mean'], cfg['im_norm_std'])(image)
    return image.unsqueeze(0)


def tensor_to_image(tensor):
    """
    transforms a tensor to an image
    Args:
        tensor: tensor of size (1, 3, x, y)
    Returns: tensor of size(x, y, 3)
    """
    x, y = tensor.size()[-2:]
    a, b, c = tensor[0]
    return torch.cat(
        (a.reshape(x, y, 1), b.reshape(x, y, 1), c.reshape(x, y, 1)), 2)


def from_dense_to_sparse_object_mask(dense_mask, boxes=None):
    """Convert a histomicsTK object segmentation-style mask,
    where first channel encodes label (ignored), and the product of the
    second and third channels encodes the object (nucleus) ids... into
    a sparse mask where each object occupies one channel (binary).

    See: https://github.com/DigitalSlideArchive/HistomicsTK/blob/master/ ...
     .. histomicstk/annotations_and_masks/annotations_to_object_mask_handler.py
    """
    dense_mask = np.uint8(dense_mask)
    labels_channel = dense_mask[..., 0]
    dense_mask = dense_mask[..., 1] * dense_mask[..., 2]

    if boxes is not None:
        # Use given bounding boxes as locations of nuclei
        # FIXME: the following assumes that the center of the
        #  bounding box is always contained within the nucleus .. i.e.
        #  that the nucleus is not weirdly concave or has holes. This is
        #  a reasonable assumption that will almost always hold true
        xmins, ymins, xmaxs, ymaxs = np.split(boxes, 4, axis=1)
        xs = xmins + ((xmaxs - xmins) // 2)
        ys = ymins + ((ymaxs - ymins) // 2)
        obj_ids = dense_mask[ys[:, 0], xs[:, 0]]
        labels = labels_channel[ys[:, 0], xs[:, 0]].tolist()
    else:
        # get object ids corresponding to nuclei from the mask itself
        obj_ids = np.unique(dense_mask)
        obj_ids = obj_ids[1:]  # remove background (zero)

        # mTODO?: get labels from mask itself
        labels = []  # for now

    # split the flat mask into a set of binary masks
    sparse_mask = dense_mask == obj_ids[:, None, None]

    return sparse_mask, labels


def _add_object_to_roi(object_mask, roi, xmin, ymin, xmax, ymax, code):
    small_object_mask = object_mask[ymin:ymax, xmin:xmax]
    patch = roi[ymin:ymax, xmin:xmax]
    patch[small_object_mask > 0] = code
    roi[ymin:ymax, xmin:xmax] = patch.copy()
    return roi


def from_sparse_to_dense_object_mask(
        sparse_mask, labels=None, min_side=None, max_side=None):
    """Convert a sparse mask where each object occupies one channel (binary),
    into a histomicsTK object segmentation-style mask, where first channel
    encodes label (ignored), and the product of the second and third channels
    encodes the object (nucleus) ids.

    See: https://github.com/DigitalSlideArchive/HistomicsTK/blob/master/ ...
     .. histomicstk/annotations_and_masks/annotations_to_object_mask_handler.py
    """
    labels = [1] * sparse_mask.shape[0] if labels is None else labels

    # init channels
    labels_channel = np.zeros(
        (sparse_mask.shape[1], sparse_mask.shape[2]), dtype=np.uint8)
    objects_channel1 = labels_channel.copy()
    objects_channel2 = labels_channel.copy()

    # unique combinations of number to be multiplied (second & third channel)
    # to be able to reconstruct the object ID when image is re-read
    object_code_comb = combinations(range(1, 256), 2)

    # this will map the instance code (product of 2nd and 3rd channel) to label
    labels_map = {}

    for objno in range(sparse_mask.shape[0]):
        object_code = next(object_code_comb)
        object_mask = sparse_mask[objno, ...]

        if np.sum(object_mask) == 0:
            continue

        pos = np.where(object_mask)
        locs = {
            'xmin': np.min(pos[1]),
            'xmax': np.max(pos[1]),
            'ymin': np.min(pos[0]),
            'ymax': np.max(pos[0]),
        }
        if (min_side is not None) and (
            locs['xmax'] - locs['xmin'] < min_side
            or (locs['ymax'] - locs['ymin'] < min_side)
        ):
            continue
        if (max_side is not None) and (
            locs['xmax'] - locs['xmin'] > max_side
            or (locs['ymax'] - locs['ymin'] > max_side)
        ):
            continue

        # add to label channel
        labels_channel = _add_object_to_roi(
            object_mask=object_mask, roi=labels_channel,
            code=labels[objno], **locs)

        # add to first objects channel
        objects_channel1 = _add_object_to_roi(
            object_mask=object_mask, roi=objects_channel1,
            code=object_code[0], **locs)

        # add to second objects channel
        objects_channel2 = _add_object_to_roi(
            object_mask=object_mask, roi=objects_channel2,
            code=object_code[1], **locs)

        labels_map[object_code[0] * object_code[1]] = labels[objno]

    dense_mask = np.uint8(np.concatenate((
        labels_channel[..., None],
        objects_channel1[..., None],
        objects_channel2[..., None],
    ), -1))

    return dense_mask, labels_map


def parse_sparse_mask_for_use(
        sparse_mask, labels: list = None, rgtcodes: dict = None,
        min_bbox_side=None, max_bbox_side=None):
    """Parse sparse mask for visualization and pushing to histomicsUI etc.

    Parameters
    ----------
    sparse_mask (np.array): n_objects, m, n
    labels (list): labels corresponding to the channels
    rgtcodes (dict): keys are integer ground truth codes, values are the
        histomicstk style for this label (group, lineColor, ...)

    Returns
    -------
    np.array: dense mask where first channel is label (semantic segmentation),
        while product of second and third channels is nucleus id
    dict: keys are indivisual nucleus ids, values are labels of nuclei
    DataFrame: each row is a contour. Histomicstk style.

    """
    if labels is None:
        labels = [1] * sparse_mask.shape[0]
        rgtcodes = {1: {'group': 'nucleus', 'color': 'rgb(255,255,0)'}}

    rgtcodes = DefaultAnnotationStyles.rgtcodes_dict if rgtcodes is None else \
        rgtcodes

    # "condense" masks
    dense_mask, labels_map = from_sparse_to_dense_object_mask(
        sparse_mask=sparse_mask, labels=labels,
        min_side=min_bbox_side, max_side=max_bbox_side)
    nids_mask = np.float32(dense_mask)
    nids_mask = nids_mask[..., 1] * nids_mask[..., 2]

    # extract contours from condensed mask
    contours_df = get_contours_from_mask(
        MASK=nids_mask,
        GTCodes_df=DataFrame.from_records(data=[
            {
                'group': rgtcodes[label]['group'],
                'GT_code': instanceid,
                'color': rgtcodes[label]['color']
            }
            for instanceid, label in labels_map.items()
        ]),
        MIN_SIZE=1 if min_bbox_side is None else min_bbox_side,
        get_roi_contour=False,
    )

    return dense_mask, labels_map, contours_df
