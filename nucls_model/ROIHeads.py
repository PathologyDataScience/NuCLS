from collections import OrderedDict

import torch
import torchvision

import torch.nn.functional as F
from torch import nn, Tensor

from torchvision.ops import boxes as box_ops

from torchvision.ops import roi_align

from torchvision.models.detection import _utils as det_utils

from torch.jit.annotations import Optional, List, Dict, Tuple
import numpy as np

from TorchUtils import tensor_isin
from nucls_model.torchvision_detection_utils.transforms import \
    remove_degenerate_bboxes


@torch.jit._script_if_tracing
def global_nms(boxes, scores, iou_threshold):
    # type: (Tensor, Tensor, float) -> Tensor
    """
    Performs non-maximum suppression globally (regardless of category).

    Parameters
    ----------
    boxes : Tensor[N, 4]
        boxes where NMS will be performed. They
        are expected to be in (x1, y1, x2, y2) format
    scores : Tensor[N]
        scores for each one of the boxes
    iou_threshold : float
        discards all overlapping boxes
        with IoU > iou_threshold

    Returns
    -------
    keep : Tensor
        int64 tensor with the indices of
        the elements that have been kept by NMS, sorted
        in decreasing order of scores
    """
    if boxes.numel() == 0:
        return torch.empty((0,), dtype=torch.int64, device=boxes.device)
    else:
        keep = box_ops.nms(boxes, scores, iou_threshold)
        return keep


# noinspection LongLine,PyTypeHints
def fastrcnn_loss(
        class_logits, box_regression, labels, regression_targets,
        ignore_label=None, batched_nms=True):
    # type: (Tensor, Tensor, List[Tensor], List[Tensor]) -> Tuple[Tensor, Tensor]
    """
    Computes the loss for Faster R-CNN.

    Arguments:
        class_logits (Tensor)
        box_regression (Tensor)
        labels (list[BoxList])
        regression_targets (Tensor)
        ignore_label (int)
        batched_nms (bool)

    Returns:
        classification_loss (Tensor)
        box_loss (Tensor)
    """

    labels = torch.cat(labels, dim=0)
    regression_targets = torch.cat(regression_targets, dim=0)

    # Mohamed: Ignore certain nuclei from classification loss.
    # Do not to modify in-place since labels are used later here!!
    clkeep = labels != ignore_label
    classification_loss = F.cross_entropy(class_logits[clkeep], labels[clkeep])

    # get indices that correspond to the regression targets for
    # the corresponding ground truth labels, to be used with
    # advanced indexing
    sampled_pos_inds_subset = torch.nonzero(labels > 0).squeeze(1)
    if batched_nms:
        labels_pos = labels[sampled_pos_inds_subset]
    else:
        # Mohamed: only two classes (bckgrnd, frgrnd) for box regression
        labs = 0 + labels
        labs[labs > 1] = 1
        labels_pos = labs[sampled_pos_inds_subset]
    N, num_classes = class_logits.shape
    box_regression = box_regression.reshape(N, -1, 4)

    box_loss = det_utils.smooth_l1_loss(
        box_regression[sampled_pos_inds_subset, labels_pos],
        regression_targets[sampled_pos_inds_subset],
        beta=1 / 9,
        size_average=False,
    )
    box_loss = box_loss / labels.numel()

    return classification_loss, box_loss


def maskrcnn_inference(x, labels):
    # type: (Tensor, List[Tensor]) -> List[Tensor]
    """
    From the results of the CNN, post process the masks
    by taking the mask corresponding to the class with max
    probability (which are of fixed size and directly output
    by the CNN) and return the masks in the mask field of the BoxList.

    Arguments:
        x (Tensor): the mask logits
        labels (list[BoxList]): bounding boxes that are used as
            reference, one for ech image

    Returns:
        results (list[BoxList]): one BoxList for each image, containing
            the extra field mask
    """
    mask_prob = x.sigmoid()

    # select masks corresponding to the predicted classes
    num_masks = x.shape[0]
    boxes_per_image = [label.shape[0] for label in labels]
    labels = torch.cat(labels)
    index = torch.arange(num_masks, device=labels.device)
    mask_prob = mask_prob[index, labels][:, None]
    mask_prob = mask_prob.split(boxes_per_image, dim=0)

    return mask_prob


def project_masks_on_boxes(gt_masks, boxes, matched_idxs, M):
    # type: (Tensor, Tensor, Tensor, int) -> Tensor
    """
    Given segmentation masks and the bounding boxes corresponding
    to the location of the masks in the image, this function
    crops and resizes the masks in the position defined by the
    boxes. This prepares the masks for them to be fed to the
    loss computation as the targets.
    """
    matched_idxs = matched_idxs.to(boxes)
    rois = torch.cat([matched_idxs[:, None], boxes], dim=1)
    gt_masks = gt_masks[:, None].to(rois)
    return roi_align(gt_masks, rois, (M, M), 1.)[:, 0]


# noinspection LongLine,PyTypeHints
def maskrcnn_loss(
        mask_logits, proposals, gt_masks, gt_labels, mask_matched_idxs,
        gt_ismask=None):
    # type: (Tensor, List[Tensor], List[Tensor], List[Tensor], List[Tensor]) -> Tensor
    """
    Arguments:
        mask_logits (Tensor)
        proposals (list[BoxList])

    Return:
        mask_loss (Tensor): scalar tensor containing the loss

    Parameters
    ----------
    gt_ismask
    mask_matched_idxs
    gt_labels
    proposals
    mask_logits
    gt_masks
    """

    discretization_size = mask_logits.shape[-1]
    labels = [gt_label[idxs] for gt_label, idxs in zip(gt_labels, mask_matched_idxs)]
    mask_targets = [
        project_masks_on_boxes(m, p, i, discretization_size)
        for m, p, i in zip(gt_masks, proposals, mask_matched_idxs)
    ]

    labels = torch.cat(labels, dim=0)
    mask_targets = torch.cat(mask_targets, dim=0)

    # Mohamed: added this to disregard nuclei without masks (bboxes)
    if gt_ismask is not None:
        keep_idxs = [torch.where(t > 0)[0] for t in gt_ismask]
        keep_list = [
            tensor_isin(idxs, keep_idx)
            for idxs, keep_idx in zip(mask_matched_idxs, keep_idxs)
        ]
        keep_array = torch.cat(keep_list).type(torch.bool)
        mask_logits = mask_logits[keep_array, ...]
        mask_targets = mask_targets[keep_array, ...]
        labels = labels[keep_array]

    # torch.mean (in binary_cross_entropy_with_logits) doesn't
    # accept empty tensors, so handle it separately
    if mask_targets.numel() == 0:
        return mask_logits.sum() * 0

    mask_loss = F.binary_cross_entropy_with_logits(
        mask_logits[torch.arange(labels.shape[0], device=labels.device), labels], mask_targets
    )
    return mask_loss


def keypoints_to_heatmap(keypoints, rois, heatmap_size):
    # type: (Tensor, Tensor, int) -> Tuple[Tensor, Tensor]
    offset_x = rois[:, 0]
    offset_y = rois[:, 1]
    scale_x = heatmap_size / (rois[:, 2] - rois[:, 0])
    scale_y = heatmap_size / (rois[:, 3] - rois[:, 1])

    offset_x = offset_x[:, None]
    offset_y = offset_y[:, None]
    scale_x = scale_x[:, None]
    scale_y = scale_y[:, None]

    x = keypoints[..., 0]
    y = keypoints[..., 1]

    x_boundary_inds = x == rois[:, 2][:, None]
    y_boundary_inds = y == rois[:, 3][:, None]

    x = (x - offset_x) * scale_x
    x = x.floor().long()
    y = (y - offset_y) * scale_y
    y = y.floor().long()

    x[x_boundary_inds] = heatmap_size - 1
    y[y_boundary_inds] = heatmap_size - 1

    valid_loc = (x >= 0) & (y >= 0) & (x < heatmap_size) & (y < heatmap_size)
    vis = keypoints[..., 2] > 0
    valid = (valid_loc & vis).long()

    lin_ind = y * heatmap_size + x
    heatmaps = lin_ind * valid

    return heatmaps, valid


# noinspection LongLine
def _onnx_heatmaps_to_keypoints(maps, maps_i, roi_map_width, roi_map_height,
                                widths_i, heights_i, offset_x_i, offset_y_i):
    num_keypoints = torch.scalar_tensor(maps.size(1), dtype=torch.int64)

    width_correction = widths_i / roi_map_width
    height_correction = heights_i / roi_map_height

    roi_map = F.interpolate(
        maps_i[:, None], size=(int(roi_map_height), int(roi_map_width)), mode='bicubic', align_corners=False)[:, 0]

    w = torch.scalar_tensor(roi_map.size(2), dtype=torch.int64)
    pos = roi_map.reshape(num_keypoints, -1).argmax(dim=1)

    x_int = (pos % w)
    y_int = ((pos - x_int) // w)

    x = (torch.tensor(0.5, dtype=torch.float32) + x_int.to(dtype=torch.float32)) * \
        width_correction.to(dtype=torch.float32)
    y = (torch.tensor(0.5, dtype=torch.float32) + y_int.to(dtype=torch.float32)) * \
        height_correction.to(dtype=torch.float32)

    xy_preds_i_0 = x + offset_x_i.to(dtype=torch.float32)
    xy_preds_i_1 = y + offset_y_i.to(dtype=torch.float32)
    xy_preds_i_2 = torch.ones((xy_preds_i_1.shape), dtype=torch.float32)  # noqa
    xy_preds_i = torch.stack([xy_preds_i_0.to(dtype=torch.float32),
                              xy_preds_i_1.to(dtype=torch.float32),
                              xy_preds_i_2.to(dtype=torch.float32)], 0)

    # mTODO: simplify when indexing without rank will be supported by ONNX
    base = num_keypoints * num_keypoints + num_keypoints + 1
    ind = torch.arange(num_keypoints)
    ind = ind.to(dtype=torch.int64) * base
    end_scores_i = roi_map.index_select(1, y_int.to(dtype=torch.int64)) \
        .index_select(2, x_int.to(dtype=torch.int64)).view(-1).index_select(0, ind.to(dtype=torch.int64))

    return xy_preds_i, end_scores_i


# noinspection LongLine
@torch.jit._script_if_tracing
def _onnx_heatmaps_to_keypoints_loop(maps, rois, widths_ceil, heights_ceil,
                                     widths, heights, offset_x, offset_y, num_keypoints):
    xy_preds = torch.zeros((0, 3, int(num_keypoints)), dtype=torch.float32, device=maps.device)
    end_scores = torch.zeros((0, int(num_keypoints)), dtype=torch.float32, device=maps.device)

    for i in range(int(rois.size(0))):
        xy_preds_i, end_scores_i = _onnx_heatmaps_to_keypoints(maps, maps[i],
                                                               widths_ceil[i], heights_ceil[i],
                                                               widths[i], heights[i],
                                                               offset_x[i], offset_y[i])
        xy_preds = torch.cat((xy_preds.to(dtype=torch.float32),
                              xy_preds_i.unsqueeze(0).to(dtype=torch.float32)), 0)
        end_scores = torch.cat((end_scores.to(dtype=torch.float32),
                                end_scores_i.to(dtype=torch.float32).unsqueeze(0)), 0)
    return xy_preds, end_scores


# noinspection LongLine
def heatmaps_to_keypoints(maps, rois):
    """Extract predicted keypoint locations from heatmaps. Output has shape
    (#rois, 4, #keypoints) with the 4 rows corresponding to (x, y, logit, prob)
    for each keypoint.
    """
    # This function converts a discrete image coordinate in a HEATMAP_SIZE x
    # HEATMAP_SIZE image to a continuous keypoint coordinate. We maintain
    # consistency with keypoints_to_heatmap_labels by using the conversion from
    # Heckbert 1990: c = d + 0.5, where d is a discrete coordinate and c is a
    # continuous coordinate.
    offset_x = rois[:, 0]
    offset_y = rois[:, 1]

    widths = rois[:, 2] - rois[:, 0]
    heights = rois[:, 3] - rois[:, 1]
    widths = widths.clamp(min=1)
    heights = heights.clamp(min=1)
    widths_ceil = widths.ceil()
    heights_ceil = heights.ceil()

    num_keypoints = maps.shape[1]

    if torchvision._is_tracing():
        xy_preds, end_scores = _onnx_heatmaps_to_keypoints_loop(maps, rois,
                                                                widths_ceil, heights_ceil, widths, heights,
                                                                offset_x, offset_y,
                                                                torch.scalar_tensor(num_keypoints, dtype=torch.int64))
        return xy_preds.permute(0, 2, 1), end_scores

    xy_preds = torch.zeros((len(rois), 3, num_keypoints), dtype=torch.float32, device=maps.device)
    end_scores = torch.zeros((len(rois), num_keypoints), dtype=torch.float32, device=maps.device)
    for i in range(len(rois)):
        roi_map_width = int(widths_ceil[i].item())
        roi_map_height = int(heights_ceil[i].item())
        width_correction = widths[i] / roi_map_width
        height_correction = heights[i] / roi_map_height
        roi_map = F.interpolate(
            maps[i][:, None], size=(roi_map_height, roi_map_width), mode='bicubic', align_corners=False)[:, 0]
        # roi_map_probs = scores_to_probs(roi_map.copy())
        w = roi_map.shape[2]
        pos = roi_map.reshape(num_keypoints, -1).argmax(dim=1)

        x_int = pos % w
        y_int = (pos - x_int) // w
        # assert (roi_map_probs[k, y_int, x_int] ==
        #         roi_map_probs[k, :, :].max())
        x = (x_int.float() + 0.5) * width_correction
        y = (y_int.float() + 0.5) * height_correction
        xy_preds[i, 0, :] = x + offset_x[i]
        xy_preds[i, 1, :] = y + offset_y[i]
        xy_preds[i, 2, :] = 1
        end_scores[i, :] = roi_map[torch.arange(num_keypoints), y_int, x_int]

    return xy_preds.permute(0, 2, 1), end_scores


# noinspection LongLine
def keypointrcnn_loss(keypoint_logits, proposals, gt_keypoints, keypoint_matched_idxs):
    # type: (Tensor, List[Tensor], List[Tensor], List[Tensor]) -> Tensor
    N, K, H, W = keypoint_logits.shape
    assert H == W
    discretization_size = H
    heatmaps = []
    valid = []
    for proposals_per_image, gt_kp_in_image, midx in zip(proposals, gt_keypoints, keypoint_matched_idxs):
        kp = gt_kp_in_image[midx]
        heatmaps_per_image, valid_per_image = keypoints_to_heatmap(
            kp, proposals_per_image, discretization_size
        )
        heatmaps.append(heatmaps_per_image.view(-1))
        valid.append(valid_per_image.view(-1))

    keypoint_targets = torch.cat(heatmaps, dim=0)
    valid = torch.cat(valid, dim=0).to(dtype=torch.uint8)
    valid = torch.nonzero(valid).squeeze(1)

    # torch.mean (in binary_cross_entropy_with_logits) does'nt
    # accept empty tensors, so handle it sepaartely
    if keypoint_targets.numel() == 0 or len(valid) == 0:
        return keypoint_logits.sum() * 0

    keypoint_logits = keypoint_logits.view(N * K, H * W)

    keypoint_loss = F.cross_entropy(keypoint_logits[valid], keypoint_targets[valid])
    return keypoint_loss


# noinspection LongLine
def keypointrcnn_inference(x, boxes):
    # type: (Tensor, List[Tensor]) -> Tuple[List[Tensor], List[Tensor]]
    kp_probs = []
    kp_scores = []

    boxes_per_image = [box.size(0) for box in boxes]  # noqa
    x2 = x.split(boxes_per_image, dim=0)

    for xx, bb in zip(x2, boxes):
        kp_prob, scores = heatmaps_to_keypoints(xx, bb)
        kp_probs.append(kp_prob)
        kp_scores.append(scores)

    return kp_probs, kp_scores


# noinspection LongLine,DuplicatedCode
def _onnx_expand_boxes(boxes, scale):
    # type: (Tensor, float) -> Tensor
    w_half = (boxes[:, 2] - boxes[:, 0]) * .5
    h_half = (boxes[:, 3] - boxes[:, 1]) * .5
    x_c = (boxes[:, 2] + boxes[:, 0]) * .5
    y_c = (boxes[:, 3] + boxes[:, 1]) * .5

    w_half = w_half.to(dtype=torch.float32) * scale
    h_half = h_half.to(dtype=torch.float32) * scale

    boxes_exp0 = x_c - w_half
    boxes_exp1 = y_c - h_half
    boxes_exp2 = x_c + w_half
    boxes_exp3 = y_c + h_half
    boxes_exp = torch.stack((boxes_exp0, boxes_exp1, boxes_exp2, boxes_exp3), 1)
    return boxes_exp


# the next two functions should be merged inside Masker
# but are kept here for the moment while we need them
# temporarily for paste_mask_in_image
# noinspection DuplicatedCode
def expand_boxes(boxes, scale):
    # type: (Tensor, float) -> Tensor
    if torchvision._is_tracing():
        return _onnx_expand_boxes(boxes, scale)
    w_half = (boxes[:, 2] - boxes[:, 0]) * .5
    h_half = (boxes[:, 3] - boxes[:, 1]) * .5
    x_c = (boxes[:, 2] + boxes[:, 0]) * .5
    y_c = (boxes[:, 3] + boxes[:, 1]) * .5

    w_half *= scale
    h_half *= scale

    boxes_exp = torch.zeros_like(boxes)
    boxes_exp[:, 0] = x_c - w_half
    boxes_exp[:, 2] = x_c + w_half
    boxes_exp[:, 1] = y_c - h_half
    boxes_exp[:, 3] = y_c + h_half
    return boxes_exp


# noinspection LongLine
@torch.jit.unused
def expand_masks_tracing_scale(M, padding):
    # type: (int, int) -> float
    return torch.tensor(M + 2 * padding).to(torch.float32) / torch.tensor(M).to(torch.float32)


# noinspection LongLine
def expand_masks(mask, padding):
    # type: (Tensor, int) -> Tuple[Tensor, float]
    M = mask.shape[-1]
    if torch._C._get_tracing_state():  # could not import is_tracing(), not sure why
        scale = expand_masks_tracing_scale(M, padding)
    else:
        scale = float(M + 2 * padding) / M
    padded_mask = torch.nn.functional.pad(mask, (padding,) * 4)
    return padded_mask, scale


# noinspection LongLine
def paste_mask_in_image(mask, box, im_h, im_w, im_mask=None, ocode=None):
    # type: (Tensor, Tensor, int, int) -> Tensor
    TO_REMOVE = 1
    w = int(box[2] - box[0] + TO_REMOVE)
    h = int(box[3] - box[1] + TO_REMOVE)
    w = max(w, 1)
    h = max(h, 1)

    # Set shape to [batchxCxHxW]
    mask = mask.expand((1, 1, -1, -1))

    # Resize mask
    mask = F.interpolate(mask, size=(h, w), mode='bilinear', align_corners=False)
    mask = mask[0][0]

    x_0 = max(box[0], 0)
    x_1 = min(box[2] + 1, im_w)
    y_0 = max(box[1], 0)
    y_1 = min(box[3] + 1, im_h)

    if im_mask is None:
        # sparse mask (only one object per channel)
        im_mask = torch.zeros((im_h, im_w), dtype=mask.dtype, device=mask.device)
        im_mask[y_0:y_1, x_0:x_1] = mask[
            (y_0 - box[1]):(y_1 - box[1]), (x_0 - box[0]):(x_1 - box[0])
        ]
    else:
        # Mohamed: dense mask (just one channel, where code represents object)
        # IMPORTANT NOTE: this means we threshold probabilities.
        patch = im_mask[y_0:y_1, x_0:x_1]
        omask = mask[
            (y_0 - box[1]):(y_1 - box[1]), (x_0 - box[0]):(x_1 - box[0])
        ]
        patch[omask > 0.5] = ocode

    return im_mask


# noinspection LongLine
def _onnx_paste_mask_in_image(mask, box, im_h, im_w):
    one = torch.ones(1, dtype=torch.int64)
    zero = torch.zeros(1, dtype=torch.int64)

    w = (box[2] - box[0] + one)
    h = (box[3] - box[1] + one)
    w = torch.max(torch.cat((w, one)))
    h = torch.max(torch.cat((h, one)))

    # Set shape to [batchxCxHxW]
    mask = mask.expand((1, 1, mask.size(0), mask.size(1)))

    # Resize mask
    mask = F.interpolate(mask, size=(int(h), int(w)), mode='bilinear', align_corners=False)
    mask = mask[0][0]

    x_0 = torch.max(torch.cat((box[0].unsqueeze(0), zero)))
    x_1 = torch.min(torch.cat((box[2].unsqueeze(0) + one, im_w.unsqueeze(0))))
    y_0 = torch.max(torch.cat((box[1].unsqueeze(0), zero)))
    y_1 = torch.min(torch.cat((box[3].unsqueeze(0) + one, im_h.unsqueeze(0))))

    unpaded_im_mask = mask[(y_0 - box[1]):(y_1 - box[1]),
                           (x_0 - box[0]):(x_1 - box[0])]

    # mTODO : replace below with a dynamic padding when support is added in ONNX

    # pad y
    zeros_y0 = torch.zeros(y_0, unpaded_im_mask.size(1))
    zeros_y1 = torch.zeros(im_h - y_1, unpaded_im_mask.size(1))
    concat_0 = torch.cat((zeros_y0,
                          unpaded_im_mask.to(dtype=torch.float32),
                          zeros_y1), 0)[0:im_h, :]
    # pad x
    zeros_x0 = torch.zeros(concat_0.size(0), x_0)
    zeros_x1 = torch.zeros(concat_0.size(0), im_w - x_1)
    im_mask = torch.cat((zeros_x0,
                         concat_0,
                         zeros_x1), 1)[:, :im_w]
    return im_mask


@torch.jit._script_if_tracing
def _onnx_paste_masks_in_image_loop(masks, boxes, im_h, im_w):
    res_append = torch.zeros(0, im_h, im_w)
    for i in range(masks.size(0)):
        mask_res = _onnx_paste_mask_in_image(masks[i][0], boxes[i], im_h, im_w)
        mask_res = mask_res.unsqueeze(0)
        res_append = torch.cat((res_append, mask_res))
    return res_append


# noinspection LongLine
def paste_masks_in_image(masks, boxes, img_shape, padding=1):
    # type: (Tensor, Tensor, Tuple[int, int], int) -> Tensor
    masks, scale = expand_masks(masks, padding=padding)
    boxes = expand_boxes(boxes, scale).to(dtype=torch.int64)
    im_h, im_w = img_shape

    if torchvision._is_tracing():
        return _onnx_paste_masks_in_image_loop(masks, boxes,
                                               torch.scalar_tensor(im_h, dtype=torch.int64),
                                               torch.scalar_tensor(im_w, dtype=torch.int64))[:, None]
    res = [
        paste_mask_in_image(m[0], b, im_h, im_w)
        for m, b in zip(masks, boxes)
    ]
    if len(res) > 0:
        ret = torch.stack(res, dim=0)[:, None]
    else:
        ret = masks.new_empty((0, 1, im_h, im_w))
    return ret

# Mohamed: dense mask (code represents object)
def paste_and_densify_masks_in_image(masks, boxes, img_shape, padding=1):
    # type: (Tensor, Tensor, Tuple[int, int], int) -> Tensor

    masks, scale = expand_masks(masks, padding=padding)
    boxes = expand_boxes(boxes, scale).to(dtype=torch.int64)
    im_h, im_w = img_shape

    if torchvision._is_tracing():
        raise NotImplementedError(
            "I didn't support tracing yet. See paste_masks_in_image()."
        )

    # IMPORTANT NOTE: in the dense mask, each object id encoded by a code
    # that corresponds to (1 + idx), where idx is the index of the object
    # in the labels/bbox tensors. For example, where im_mask == 5 corresponds
    # to the nucleus whose label is in the 4th index in the labels tensor

    # Find the order in which to overlay objects in the mask, from big to small
    # This prevents a big object from covering a small one. Noe, however, that
    # 1. This is still a "lossy" process, but it saves up on speed and makes
    #    the memory requirement less dependent on the no of objects
    # 2. We maintain the object codes' correspondence to (1 + idx)
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    idxs = torch.argsort(areas, descending=True)

    im_mask = torch.zeros((im_h, im_w), dtype=torch.int64, device=masks.device)
    for idx in idxs:
        im_mask = paste_mask_in_image(
            mask=masks[idx][0], box=boxes[idx], im_h=im_h, im_w=im_w,
            im_mask=im_mask, ocode=idx + 1)

    return im_mask

# noinspection LongLine
class RoIHeads(torch.nn.Module):
    __annotations__ = {
        'box_coder': det_utils.BoxCoder,
        'proposal_matcher': det_utils.Matcher,
        'fg_bg_sampler': det_utils.BalancedPositiveNegativeSampler,
    }

    def __init__(self,
                 box_roi_pool,
                 box_head,
                 box_predictor,
                 # Faster R-CNN training
                 fg_iou_thresh, bg_iou_thresh,
                 batch_size_per_image, positive_fraction,
                 bbox_reg_weights,
                 # Faster R-CNN inference
                 score_thresh,
                 nms_thresh,
                 detections_per_img,
                 # Mask
                 mask_roi_pool=None,
                 mask_head=None,
                 mask_predictor=None,
                 keypoint_roi_pool=None,
                 keypoint_head=None,
                 keypoint_predictor=None,
                 # added by Mohamed
                 batched_nms=True,  # inference nms independently per class?
                 indep_classif_boxes=False,
                 classification_bbox_size=None,  # float
                 cconvhead=None,  # extra conv layers for classification
                 sattention_head=None,  # nuclei are aware of each other
                 ignore_label: int = None,  # label to ignore in classif. loss
                 ):
        super(RoIHeads, self).__init__()

        if indep_classif_boxes:
            assert classification_bbox_size is not None

        self.box_similarity = box_ops.box_iou
        # assign ground-truth boxes for each proposal
        self.proposal_matcher = det_utils.Matcher(
            fg_iou_thresh,
            bg_iou_thresh,
            allow_low_quality_matches=False)

        self.fg_bg_sampler = det_utils.BalancedPositiveNegativeSampler(
            batch_size_per_image,
            positive_fraction)

        if bbox_reg_weights is None:
            bbox_reg_weights = (10., 10., 5., 5.)
        self.box_coder = det_utils.BoxCoder(bbox_reg_weights)

        self.box_roi_pool = box_roi_pool
        self.box_head = box_head
        self.box_predictor = box_predictor

        self.score_thresh = score_thresh
        self.nms_thresh = nms_thresh
        self.detections_per_img = detections_per_img

        self.mask_roi_pool = mask_roi_pool
        self.mask_head = mask_head
        self.mask_predictor = mask_predictor

        self.keypoint_roi_pool = keypoint_roi_pool
        self.keypoint_head = keypoint_head
        self.keypoint_predictor = keypoint_predictor

        # added by Mohamed
        self.batched_nms = batched_nms
        self.indep_classif_boxes = indep_classif_boxes
        self.classification_bbox_size = classification_bbox_size
        self.halfclbox = self.classification_bbox_size / 2 if \
            classification_bbox_size is not None else None
        self.cconvhead = cconvhead
        self.sattention_head = sattention_head
        self.ignore_label = ignore_label

    def has_mask(self):
        if self.mask_roi_pool is None:
            return False
        if self.mask_head is None:
            return False
        if self.mask_predictor is None:
            return False
        return True

    def has_keypoint(self):
        if self.keypoint_roi_pool is None:
            return False
        if self.keypoint_head is None:
            return False
        if self.keypoint_predictor is None:
            return False
        return True

    def assign_targets_to_proposals(self, proposals, gt_boxes, gt_labels):
        # type: (List[Tensor], List[Tensor], List[Tensor]) -> Tuple[List[Tensor], List[Tensor]]
        matched_idxs = []
        labels = []
        for proposals_in_image, gt_boxes_in_image, gt_labels_in_image in zip(proposals, gt_boxes, gt_labels):

            if gt_boxes_in_image.numel() == 0:
                # Background image
                device = proposals_in_image.device
                clamped_matched_idxs_in_image = torch.zeros(
                    (proposals_in_image.shape[0],), dtype=torch.int64, device=device
                )
                labels_in_image = torch.zeros(
                    (proposals_in_image.shape[0],), dtype=torch.int64, device=device
                )
            else:
                #  set to self.box_similarity when https://github.com/pytorch/pytorch/issues/27495 lands
                match_quality_matrix = box_ops.box_iou(gt_boxes_in_image, proposals_in_image)
                matched_idxs_in_image = self.proposal_matcher(match_quality_matrix)

                clamped_matched_idxs_in_image = matched_idxs_in_image.clamp(min=0)

                labels_in_image = gt_labels_in_image[clamped_matched_idxs_in_image]
                labels_in_image = labels_in_image.to(dtype=torch.int64)

                # Label background (below the low threshold)
                bg_inds = matched_idxs_in_image == self.proposal_matcher.BELOW_LOW_THRESHOLD
                labels_in_image[bg_inds] = 0

                # Label ignore proposals (between low and high thresholds)
                ignore_inds = matched_idxs_in_image == self.proposal_matcher.BETWEEN_THRESHOLDS
                labels_in_image[ignore_inds] = -1  # -1 is ignored by sampler

            matched_idxs.append(clamped_matched_idxs_in_image)
            labels.append(labels_in_image)
        return matched_idxs, labels

    def subsample(self, labels):
        # type: (List[Tensor]) -> List[Tensor]
        sampled_pos_inds, sampled_neg_inds = self.fg_bg_sampler(labels)
        sampled_inds = []
        for img_idx, (pos_inds_img, neg_inds_img) in enumerate(
            zip(sampled_pos_inds, sampled_neg_inds)
        ):
            img_sampled_inds = torch.nonzero(pos_inds_img | neg_inds_img).squeeze(1)
            sampled_inds.append(img_sampled_inds)
        return sampled_inds

    # noinspection PyMethodMayBeStatic
    def add_gt_proposals(self, proposals, gt_boxes):
        # type: (List[Tensor], List[Tensor]) -> List[Tensor]
        proposals = [
            torch.cat((proposal, gt_box))
            for proposal, gt_box in zip(proposals, gt_boxes)
        ]

        return proposals

    def check_targets(self, targets):
        # type: (Optional[List[Dict[str, Tensor]]]) -> None
        assert targets is not None
        assert all(["boxes" in t for t in targets])
        assert all(["labels" in t for t in targets])
        if self.has_mask():
            assert all(["masks" in t for t in targets])

    def select_training_samples(self,
                                proposals,  # type: List[Tensor]
                                targets     # type: Optional[List[Dict[str, Tensor]]]
                                ):
        # type: (...) -> Tuple[List[Tensor], List[Tensor], List[Tensor], List[Tensor]]
        self.check_targets(targets)
        assert targets is not None
        dtype = proposals[0].dtype
        device = proposals[0].device

        gt_boxes = [t["boxes"].to(dtype) for t in targets]
        gt_labels = [t["labels"] for t in targets]

        # append ground-truth bboxes to propos
        proposals = self.add_gt_proposals(proposals, gt_boxes)

        # get matching gt indices for each proposal
        matched_idxs, labels = self.assign_targets_to_proposals(proposals, gt_boxes, gt_labels)
        # sample a fixed proportion of positive-negative proposals
        sampled_inds = self.subsample(labels)
        matched_gt_boxes = []
        num_images = len(proposals)
        for img_id in range(num_images):
            img_sampled_inds = sampled_inds[img_id]
            proposals[img_id] = proposals[img_id][img_sampled_inds]
            labels[img_id] = labels[img_id][img_sampled_inds]
            matched_idxs[img_id] = matched_idxs[img_id][img_sampled_inds]

            gt_boxes_in_image = gt_boxes[img_id]
            if gt_boxes_in_image.numel() == 0:
                gt_boxes_in_image = torch.zeros((1, 4), dtype=dtype, device=device)
            matched_gt_boxes.append(gt_boxes_in_image[matched_idxs[img_id]])

        regression_targets = self.box_coder.encode(matched_gt_boxes, proposals)
        return proposals, matched_idxs, labels, regression_targets

    def postprocess_detections_globalnms(self,
                               class_logits,    # type: Tensor
                               box_regression,  # type: Tensor
                               proposals,       # type: List[Tensor]
                               image_shapes     # type: List[Tuple[int, int]]
                               ):
        # type: (...) -> Tuple[List[Tensor], List[Tensor], List[Tensor]]
        device = class_logits.device
        hdim = class_logits.shape[0]
        num_classes = class_logits.shape[-1]

        boxes_per_image = [boxes_in_image.shape[0] for boxes_in_image in proposals]
        pred_boxes = self.box_coder.decode(box_regression, proposals)

        pred_scores = F.softmax(class_logits, -1)

        pred_boxes_list = pred_boxes.split(boxes_per_image, 0)
        pred_scores_list = pred_scores.split(boxes_per_image, 0)

        all_boxes = []
        all_objectness = []  # probability this is an object
        all_labels = []
        all_probabs = []  # probabilities for each class
        all_keptidxs = []
        for boxes, scores, image_shape in zip(pred_boxes_list, pred_scores_list, image_shapes):

            boxes = box_ops.clip_boxes_to_image(boxes, image_shape)

            # remove predictions with the background label
            boxes = boxes[:, 1, :]
            scores = scores[:, 1:]

            # we define the "objectness" as the sum of non-background scores
            objectness = scores.sum(1)

            # indices of detections that are kept nms and postprocessing
            keptidxs = torch.arange(hdim, device=device)

            # Mohamed: prevent (set as zero prob.) ignore_label prediction
            if self.ignore_label is not None:
                ignore = [j == self.ignore_label for j in range(1, num_classes)]
                scores[:, ignore] = 0.

            # remove low scoring boxes (i.e. low "objectness").
            inds = torch.nonzero(objectness > self.score_thresh).squeeze(1)
            boxes, scores, objectness, keptidxs = boxes[inds], scores[inds], objectness[inds], keptidxs[inds]

            # remove empty boxes
            keep = box_ops.remove_small_boxes(boxes, min_size=1e-2)
            boxes, scores, objectness, keptidxs = boxes[keep], scores[keep], objectness[keep], keptidxs[keep]

            # global nms, regardless of class
            keep = global_nms(boxes=boxes, scores=objectness, iou_threshold=self.nms_thresh)

            # keep only topk scoring predictions
            keep = keep[:self.detections_per_img]
            boxes, scores, objectness, keptidxs = boxes[keep], scores[keep], objectness[keep], keptidxs[keep]

            if boxes.shape[0] > 0:
                # make sure probabilities add to 1, while preserving the
                # ignore_label columns as zero, so we divide by total instead
                # of using softmax
                scores = scores / scores.sum(1)[:, None]

                # label is argmax, just like ordinary classification tasks. Keep
                # in mind that we got rid of the background channel, so add 1
                labels = torch.argmax(scores, dim=1) + 1
            else:
                # empty tensor; nothing was left after filtering out junk
                labels = scores[:, 0].type(torch.int64)

            all_boxes.append(boxes)
            all_objectness.append(objectness)
            all_labels.append(labels)
            all_probabs.append(scores)
            all_keptidxs.append(keptidxs)

        return all_boxes, all_objectness, all_labels, all_probabs, all_keptidxs


    def postprocess_detections(self,
                               class_logits,    # type: Tensor
                               box_regression,  # type: Tensor
                               proposals,       # type: List[Tensor]
                               image_shapes     # type: List[Tuple[int, int]]
                               ):
        # type: (...) -> Tuple[List[Tensor], List[Tensor], List[Tensor]]
        device = class_logits.device
        num_classes = class_logits.shape[-1]

        boxes_per_image = [boxes_in_image.shape[0] for boxes_in_image in proposals]
        pred_boxes = self.box_coder.decode(box_regression, proposals)

        pred_scores = F.softmax(class_logits, -1)

        pred_boxes_list = pred_boxes.split(boxes_per_image, 0)
        pred_scores_list = pred_scores.split(boxes_per_image, 0)

        all_boxes = []
        all_scores = []
        all_labels = []
        for boxes, scores, image_shape in zip(pred_boxes_list, pred_scores_list, image_shapes):
            boxes = box_ops.clip_boxes_to_image(boxes, image_shape)

            # create labels for each prediction
            labels = torch.arange(num_classes, device=device)
            labels = labels.view(1, -1).expand_as(scores)

            # remove predictions with the background label
            boxes = boxes[:, 1:]
            scores = scores[:, 1:]
            labels = labels[:, 1:]

            # Mohamed: remove self.ignore_label predictions
            if self.ignore_label is not None:
                keep = [j != self.ignore_label for j in range(1, num_classes)]
                boxes = boxes[:, keep]
                scores = scores[:, keep]
                labels = labels[:, keep]

            # batch everything, by making every class prediction be a separate instance
            boxes = boxes.reshape(-1, 4)
            scores = scores.reshape(-1)
            labels = labels.reshape(-1)

            # remove low scoring boxes
            inds = torch.nonzero(scores > self.score_thresh).squeeze(1)
            boxes, scores, labels = boxes[inds], scores[inds], labels[inds]

            # remove empty boxes
            keep = box_ops.remove_small_boxes(boxes, min_size=1e-2)
            boxes, scores, labels = boxes[keep], scores[keep], labels[keep]

            # independent nms per class
            keep = box_ops.batched_nms(
                boxes=boxes, scores=scores, idxs=labels,
                iou_threshold=self.nms_thresh)

            # keep only topk scoring predictions
            keep = keep[:self.detections_per_img]
            boxes, scores, labels = boxes[keep], scores[keep], labels[keep]

            all_boxes.append(boxes)
            all_scores.append(scores)
            all_labels.append(labels)

        return all_boxes, all_scores, all_labels

    def get_classification_proposals(self, proposals, image_shapes):

        classification_proposals = []
        for pno, prop in enumerate(proposals):
            xmins, ymins, xmaxs, ymaxs = torch.chunk(prop, 4, dim=1)
            xs = xmins + (xmaxs - xmins) / 2
            ys = ymins + (ymaxs - ymins) / 2
            boxes = torch.cat(
                [xs - self.halfclbox, ys - self.halfclbox,
                 xs + self.halfclbox, ys + self.halfclbox],
                dim=1)

            # adjust boxes
            dim0, dim1 = image_shapes[pno]
            boxes, _ = remove_degenerate_bboxes(
                boxes=boxes, dim0=dim0, dim1=dim1, min_boxside=0)

            classification_proposals.append(boxes)

        return classification_proposals

    def forward(self,
                features,  # type: Dict[str, Tensor]
                proposals,  # type: List[Tensor]
                image_shapes,  # type: List[Tuple[int, int]]
                targets=None,  # type: Optional[List[Dict[str, Tensor]]]
                _just_return_probabs=False,  # type: bool
                _cprobabs=None,  # type: List[Tensor]
                _return_prepr=False,  # type: bool
                ):
        # type: (...) -> Tuple[List[Dict[str, Tensor]], Dict[str, Tensor]]
        """
        Arguments:
            features (List[Tensor])
            proposals (List[Tensor[N, 4]])
            image_shapes (List[Tuple[H, W]])
            targets (List[Dict])
            _just_return_probabs (bool): If true, just returns logits for each
                of the proposals without postprocessing. This is only applies
                if not self.training.
            _cprobabs (List[Tensor]): If given, these "past logits" (obtained
                from variations of the proposals) would be aggregated to
                the class_logits as a form of test-time augmentation. This
                obviously only applies if not self.training
            _return_prepr (bool): also return intermediate representations?
        """
        if targets is not None:
            for t in targets:
                # mTODO: https://github.com/pytorch/pytorch/issues/26731
                floating_point_types = (torch.float, torch.double, torch.half)
                assert t["boxes"].dtype in floating_point_types, 'target boxes must of float type'
                assert t["labels"].dtype == torch.int64, 'target labels must of int64 type'
                if self.has_keypoint():
                    assert t["keypoints"].dtype == torch.float32, 'target keypoints must of float type'

        if self.training:
            proposals, matched_idxs, labels, regression_targets = self.select_training_samples(proposals, targets)
        else:
            labels = None
            regression_targets = None
            matched_idxs = None

        # Added by Mohamed: Classification proposals use an
        # independent (possibly wider) region beyond object boundary
        cproposals = None if not self.indep_classif_boxes else \
            self.get_classification_proposals(proposals=proposals, image_shapes=image_shapes)

        # Mohamed: Independent, extra convolutions for classification
        cfeatures = None if self.cconvhead is None else OrderedDict(
            {k: self.cconvhead(v) for k, v in features.items()}
        )

        # roi pooling for the object bbox
        box_features = self.box_roi_pool(features, proposals, image_shapes)
        if (cproposals is not None) or (cfeatures is not None):
            cbox_features = self.box_roi_pool(
                cfeatures or features, cproposals or proposals, image_shapes)
        else:
            cbox_features = None

        # flatten
        box_features = self.box_head(box_features)
        cbox_features = self.box_head(cbox_features) \
            if cbox_features is not None else None

        # Mohamed: per-fov nuclei attentive to each other
        if self.sattention_head is not None:

            # this only applies for classification
            if cbox_features is None:
                cbox_features = 0. + box_features

            # get the start index for each group of proposals from one FOV
            fov_start_idxs = np.cumsum(
                [0] + [j.shape[0] for j in proposals[:-1]]).tolist()

            # pass through self-attention head
            cbox_features = self.sattention_head(
                cbox_features, fov_start_idxs=fov_start_idxs)

        # pass through fully-connected layers
        if cbox_features is None:
            class_logits, box_regression = self.box_predictor(box_features)
        else:
            _, box_regression = self.box_predictor(box_features, get_scores=False)
            class_logits, _ = self.box_predictor(cbox_features, get_deltas=False)

        result = torch.jit.annotate(List[Dict[str, torch.Tensor]], [])
        losses = {}
        if self.training:
            assert labels is not None and regression_targets is not None
            loss_classifier, loss_box_reg = fastrcnn_loss(
                class_logits=class_logits,
                box_regression=box_regression,
                labels=labels,
                regression_targets=regression_targets,
                ignore_label=self.ignore_label,
                batched_nms=self.batched_nms,
            )
            losses = {
                "loss_classifier": loss_classifier,
                "loss_box_reg": loss_box_reg
            }
        else:
            # Mohamed: maybe just return class probabs without postprocessing
            # Note that this INCLUDES the background class
            if _just_return_probabs:
                return F.softmax(class_logits, -1)

            # Mohamed: "past logits" (obtained from variations of proposals)
            # would be aggregated to the class_logits as a form of test-time
            # augmentation. This obviously only applies if not self.training
            if _cprobabs is not None:
                class_logits = F.softmax(class_logits, -1)
                class_logits += _cprobabs

            if self.batched_nms:
                boxes, scores, labels = self.postprocess_detections(class_logits, box_regression, proposals, image_shapes)
                probabs = [None] * len(boxes)
                keptidxs = [None] * len(boxes)
            else:
                boxes, scores, labels, probabs, keptidxs = self.postprocess_detections_globalnms(class_logits, box_regression, proposals, image_shapes)
            num_images = len(boxes)

            if _return_prepr:
                bperim = [boxes_in_image.shape[0] for boxes_in_image in proposals]
                bfeats = box_features.split(bperim, 0)
                cbfeats = None if cbox_features is None else cbox_features.split(bperim, 0)
                clogits = class_logits.split(bperim, 0)

            for i in range(num_images):
                out = {
                    "boxes": boxes[i],
                    "labels": labels[i],
                    "scores": scores[i],
                    "probabs": probabs[i],
                }
                if _return_prepr:
                    out.update({
                        "box_features": bfeats[i][keptidxs[i], :],
                        "cbox_features": None if cbfeats is None else cbfeats[i][keptidxs[i], :],
                        "clogits": clogits[i][keptidxs[i], :],
                    })
                result.append(out)

        if self.has_mask():
            mask_proposals = [p["boxes"] for p in result]
            if self.training:
                assert matched_idxs is not None
                # during training, only focus on positive boxes
                num_images = len(proposals)
                mask_proposals = []
                pos_matched_idxs = []
                for img_id in range(num_images):
                    pos = torch.nonzero(labels[img_id] > 0).squeeze(1)
                    mask_proposals.append(proposals[img_id][pos])
                    pos_matched_idxs.append(matched_idxs[img_id][pos])
            else:
                pos_matched_idxs = None

            if self.mask_roi_pool is not None:
                mask_features = self.mask_roi_pool(features, mask_proposals, image_shapes)
                mask_features = self.mask_head(mask_features)
                mask_logits = self.mask_predictor(mask_features)
            else:
                # noinspection PyUnusedLocal
                mask_logits = torch.tensor(0)
                raise Exception("Expected mask_roi_pool to be not None")

            loss_mask = {}
            if self.training:
                assert targets is not None
                assert pos_matched_idxs is not None
                assert mask_logits is not None

                # Mohamed: if not self.batched_nms, just one mask per object
                gt_labels = [0 + t["labels"] for t in targets]
                if not self.batched_nms:
                    for labs in gt_labels:
                        labs[labs > 1] = 1

                gt_masks = [t["masks"] for t in targets]
                # Mohamed: added to ignore bboxes (false masks)
                #  so that they don't contribute to the mask loss
                if 'ismask' in targets[0]:
                    gt_ismask = [t['ismask'] for t in targets]
                else:
                    gt_ismask = None
                rcnn_loss_mask = maskrcnn_loss(
                    mask_logits=mask_logits, proposals=mask_proposals,
                    gt_masks=gt_masks, gt_labels=gt_labels,
                    mask_matched_idxs=pos_matched_idxs,
                    gt_ismask=gt_ismask)
                loss_mask = {
                    "loss_mask": rcnn_loss_mask
                }
            else:
                # Mohamed: if not self.batched_nms, just one mask per object
                mask_labels = [0 + r["labels"] for r in result]
                if not self.batched_nms:
                    for labs in mask_labels:
                        labs[labs > 1] = 1
                masks_probs = maskrcnn_inference(mask_logits, mask_labels)
                for mask_prob, r in zip(masks_probs, result):
                    r["masks"] = mask_prob

            losses.update(loss_mask)

        # keep none checks in if conditional so torchscript will conditionally
        # compile each branch
        if self.keypoint_roi_pool is not None and self.keypoint_head is not None \
                and self.keypoint_predictor is not None:
            keypoint_proposals = [p["boxes"] for p in result]
            if self.training:
                # during training, only focus on positive boxes
                num_images = len(proposals)
                keypoint_proposals = []
                pos_matched_idxs = []
                assert matched_idxs is not None
                for img_id in range(num_images):
                    pos = torch.nonzero(labels[img_id] > 0).squeeze(1)
                    keypoint_proposals.append(proposals[img_id][pos])
                    pos_matched_idxs.append(matched_idxs[img_id][pos])
            else:
                pos_matched_idxs = None

            keypoint_features = self.keypoint_roi_pool(features, keypoint_proposals, image_shapes)
            keypoint_features = self.keypoint_head(keypoint_features)
            keypoint_logits = self.keypoint_predictor(keypoint_features)

            loss_keypoint = {}
            if self.training:
                assert targets is not None
                assert pos_matched_idxs is not None

                gt_keypoints = [t["keypoints"] for t in targets]
                rcnn_loss_keypoint = keypointrcnn_loss(
                    keypoint_logits, keypoint_proposals,
                    gt_keypoints, pos_matched_idxs)
                loss_keypoint = {
                    "loss_keypoint": rcnn_loss_keypoint
                }
            else:
                assert keypoint_logits is not None
                assert keypoint_proposals is not None

                keypoints_probs, kp_scores = keypointrcnn_inference(keypoint_logits, keypoint_proposals)
                for keypoint_prob, kps, r in zip(keypoints_probs, kp_scores, result):
                    r["keypoints"] = keypoint_prob
                    r["keypoints_scores"] = kps

            losses.update(loss_keypoint)

        return result, losses
