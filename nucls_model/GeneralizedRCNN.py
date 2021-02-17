# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""
Implements the Generalized R-CNN framework
"""

from collections import OrderedDict
import torch
from torch import nn
import warnings
from torch.jit.annotations import Tuple, List, Dict, Optional
from torch import Tensor


# noinspection LongLine
def _check_for_degenerate_boxes(trgts):
    if trgts is not None:
        for target_idx, target in enumerate(trgts):
            boxes = target["boxes"]
            degenerate_boxes = boxes[:, 2:] <= boxes[:, :2]
            if degenerate_boxes.any():
                # print the first degenrate box
                bb_idx = degenerate_boxes.any(dim=1).nonzero().view(-1)[0]
                degen_bb: List[float] = boxes[bb_idx].tolist()
                raise ValueError("All bounding boxes should have positive height and width."
                                 " Found invaid box {} for target at index {}."
                                 .format(degen_bb, target_idx))


# noinspection LongLine
class GeneralizedRCNN(nn.Module):
    """
    Main class for Generalized R-CNN.

    Arguments:
        backbone (nn.Module):
        rpn (nn.Module):
        roi_heads (nn.Module): takes the features + the proposals from the RPN and computes
            detections / masks from it.
        transform (nn.Module): performs the data transformation from the inputs to feed into
            the model
        n_testtime_augmentations (int): no oftest-time augmentations.
        proposal_augmenter (RpnProposalAugmenter): this is a function or class that
            can be called to obtain an augmented realization (eg random shift)
            of object proposals from the RPN output.
    """

    def __init__(
            self, backbone, rpn, roi_heads, transform,
            proposal_augmenter=None, n_testtime_augmentations=0):
        super(GeneralizedRCNN, self).__init__()
        self.transform = transform
        self.backbone = backbone
        self.rpn = rpn
        self.roi_heads = roi_heads
        # used only on torchscript mode
        self._has_warned = False
        # Mohamed: added this
        if proposal_augmenter is not None:
            assert not roi_heads.batched_nms
        self.n_testtime_augmentations = n_testtime_augmentations
        self.proposal_augmenter = proposal_augmenter

    @torch.jit.unused
    def eager_outputs(self, losses, detections):
        # type: (Dict[str, Tensor], List[Dict[str, Tensor]]) -> Tuple[Dict[str, Tensor], List[Dict[str, Tensor]]]
        if self.training:
            return losses

        return detections

    def forward(self, images, targets=None):
        # type: (List[Tensor], Optional[List[Dict[str, Tensor]]]) -> Tuple[Dict[str, Tensor], List[Dict[str, Tensor]]]
        """
        Arguments:
            images (list[Tensor]): images to be processed
            targets (list[Dict[Tensor]]): ground-truth boxes present in the image (optional)

        Returns:
            result (list[BoxList] or dict[Tensor]): the output from the model.
                During training, it returns a dict[Tensor] which contains the losses.
                During testing, it returns list[BoxList] contains additional fields
                like `scores`, `labels` and `mask` (for Mask R-CNN models).

        """
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")
        if self.training:
            assert targets is not None
            for target in targets:
                boxes = target["boxes"]
                if isinstance(boxes, torch.Tensor):
                    if len(boxes.shape) != 2 or boxes.shape[-1] != 4:
                        raise ValueError("Expected target boxes to be a tensor"
                                         "of shape [N, 4], got {:}.".format(
                                             boxes.shape))
                else:
                    raise ValueError("Expected target boxes to be of type "
                                     "Tensor, got {:}.".format(type(boxes)))

        original_image_sizes = torch.jit.annotate(List[Tuple[int, int]], [])
        for img in images:
            val = img.shape[-2:]
            assert len(val) == 2
            original_image_sizes.append((val[0], val[1]))

        images, targets = self.transform(images, targets)

        # Check for degenerate boxes
        _check_for_degenerate_boxes(targets)

        features = self.backbone(images.tensors)
        if isinstance(features, torch.Tensor):
            features = OrderedDict([('0', features)])
        proposals, proposal_losses = self.rpn(images, features, targets)

        _cprobabs = None
        if (not self.training) and (self.n_testtime_augmentations > 0):
            # Mohamed: Test-time augmentation by jittering the RPN output
            #  so that it gets projected onto the feature map multiple times
            #  resulting in slightly different outputs each time. This is a
            #  nice way of augmentation because: 1. The feature map is only
            #  extracted once, only the ROIPooling differs; 2. It augments BOTH
            #  detection and classification.
            # get augmented boxes & get class probabs without postprocessing
            for _ in range(self.n_testtime_augmentations):
                prealization = self.proposal_augmenter(
                    proposals=proposals, image_shapes=images.image_sizes)
                _cprobabs_realization = self.roi_heads(
                    features=features, proposals=prealization,
                    image_shapes=images.image_sizes, _just_return_probabs=True)
                if _cprobabs is None:
                    _cprobabs = _cprobabs_realization
                else:
                    # aggregate soft probabilities
                    _cprobabs += _cprobabs_realization
            _cprobabs = _cprobabs / self.n_testtime_augmentations

        # pass through roi head, possible aggregating probabilities obtained
        # from test-time augmentations
        detections, detector_losses = self.roi_heads(
            features=features, proposals=proposals, _cprobabs=_cprobabs,
            image_shapes=images.image_sizes, targets=targets)

        detections = self.transform.postprocess(detections, images.image_sizes, original_image_sizes)

        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)

        if torch.jit.is_scripting():
            if not self._has_warned:
                warnings.warn("RCNN always returns a (Losses, Detections) tuple in scripting")
                self._has_warned = True
            # noinspection PyRedundantParentheses
            return (losses, detections)
        else:
            return self.eager_outputs(losses, detections)
