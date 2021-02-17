from collections import OrderedDict
import warnings

import torch
from torch.jit.annotations import Tuple, List, Dict, Optional
from torch import Tensor

import sys
BASEPATH = "/home/mtageld/Desktop/cTME/"
sys.path.insert(0, BASEPATH)
from nucls_model.MaskRCNN import MaskRCNN

__all__ = ["PartialMaskRCNN"]


# noinspection LongLine
class PartialMaskRCNN(MaskRCNN):
    """
    A partial version of MaskRCNN that returns intermediate representations
    of the detected objects in inference mode.
    """
    def __init__(self, backbone, **kwargs):
        super(PartialMaskRCNN, self).__init__(backbone=backbone, **kwargs)

    def forward(self, images, targets=None):
        # type: (List[Tensor], Optional[List[Dict[str, Tensor]]]) -> Tuple[Dict[str, Tensor], List[Dict[str, Tensor]]]
        """
        Arguments:
            images (list[Tensor]): images to be processed

        Returns:
            result (list[BoxList] or dict[Tensor]): the output from the model.
                It returns list[BoxList] the contains fields
                like `scores`, `labels` and `mask` (for Mask R-CNN models).

        """
        assert not self.training, "PartialMaskRCNN only works in inference mode!"
        assert targets is None, "targets are not needed!"
        assert not self.roi_heads.batched_nms, "only global nms supported!"

        original_image_sizes = torch.jit.annotate(List[Tuple[int, int]], [])
        for img in images:
            val = img.shape[-2:]
            assert len(val) == 2
            original_image_sizes.append((val[0], val[1]))

        images, targets = self.transform(images, targets)

        features = self.backbone(images.tensors)
        if isinstance(features, torch.Tensor):
            features = OrderedDict([('0', features)])
        proposals, proposal_losses = self.rpn(images, features, targets)

        _cprobabs = None
        if self.n_testtime_augmentations > 0:
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

        # pass through roi head, possibly aggregating probabilities obtained
        # from test-time augmentations
        detections, _ = self.roi_heads(
            features=features, proposals=proposals, _cprobabs=_cprobabs,
            image_shapes=images.image_sizes, _return_prepr=True)

        detections = self.transform.postprocess(detections, images.image_sizes,
            original_image_sizes)

        if torch.jit.is_scripting():
            if not self._has_warned:
                warnings.warn(
                    "RCNN always returns a (Losses, Detections) tuple in scripting")
                self._has_warned = True
            # noinspection PyRedundantParentheses
            return ({}, detections)
        else:
            return self.eager_outputs({}, detections)

