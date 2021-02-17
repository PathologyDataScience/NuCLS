# from collections import OrderedDict

import torch
from torch import nn
import torch.nn.functional as F  # noqa
from collections import OrderedDict
from typing import List

# from torchvision.ops import misc as misc_nn_ops
from torchvision.ops import MultiScaleRoIAlign

from torchvision.models.utils import load_state_dict_from_url

# from torchvision.models.detection.generalized_rcnn import GeneralizedRCNN
# from torchvision.models.detection.roi_heads import RoIHeads
# from torchvision.models.detection.transform import GeneralizedRCNNTransform
from torchvision.models.detection.rpn import AnchorGenerator, RPNHead, \
    RegionProposalNetwork
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone

from nucls_model.GeneralizedRCNN import GeneralizedRCNN
from nucls_model.torchvision_detection_utils.Transform import \
    GeneralizedRCNNTransform
from nucls_model.ROIHeads import RoIHeads

__all__ = [
    "FasterRCNN", "fasterrcnn_resnet50_fpn",
]


# noinspection LongLine
class FasterRCNN(GeneralizedRCNN):
    """
    Implements Faster R-CNN.

    The input to the model is expected to be a list of tensors, each of shape [C, H, W], one for each
    image, and should be in 0-1 range. Different images can have different sizes.

    The behavior of the model changes depending if it is in training or evaluation mode.

    During training, the model expects both the input tensors, as well as a targets (list of dictionary),
    containing:
        - boxes (FloatTensor[N, 4]): the ground-truth boxes in [x1, y1, x2, y2] format, with values
          between 0 and H and 0 and W
        - labels (Int64Tensor[N]): the class label for each ground-truth box

    The model returns a Dict[Tensor] during training, containing the classification and regression
    losses for both the RPN and the R-CNN.

    During inference, the model requires only the input tensors, and returns the post-processed
    predictions as a List[Dict[Tensor]], one for each input image. The fields of the Dict are as
    follows:
        - boxes (FloatTensor[N, 4]): the predicted boxes in [x1, y1, x2, y2] format, with values between
          0 and H and 0 and W
        - labels (Int64Tensor[N]): the predicted labels for each image
        - scores (Tensor[N]): the scores or each prediction

    Arguments:
        backbone (nn.Module): the network used to compute the features for the model.
            It should contain a out_channels attribute, which indicates the number of output
            channels that each feature map has (and it should be the same for all feature maps).
            The backbone should return a single Tensor or and OrderedDict[Tensor].
        num_classes (int): number of output classes of the model (including the background).
            If box_predictor is specified, num_classes should be None.
        scale_factor (float): factor by which image is to be rescaled before feeding it to the backbone
        scale_factor_jitter (float): a random float less than this amount will be added or subtracted
            from the scale_factor during trainign for scale augmentation
        image_mean (Tuple[float, float, float]): mean values used for input normalization.
            They are generally the mean values of the dataset on which the backbone has been trained
            on
        image_std (Tuple[float, float, float]): std values used for input normalization.
            They are generally the std values of the dataset on which the backbone has been trained on
        rpn_anchor_generator (AnchorGenerator): module that generates the anchors for a set of feature
            maps.
        rpn_head (nn.Module): module that computes the objectness and regression deltas from the RPN
        rpn_pre_nms_top_n_train (int): number of proposals to keep before applying NMS during training
        rpn_pre_nms_top_n_test (int): number of proposals to keep before applying NMS during testing
        rpn_post_nms_top_n_train (int): number of proposals to keep after applying NMS during training
        rpn_post_nms_top_n_test (int): number of proposals to keep after applying NMS during testing
        rpn_nms_thresh (float): NMS threshold used for postprocessing the RPN proposals
        rpn_fg_iou_thresh (float): minimum IoU between the anchor and the GT box so that they can be
            considered as positive during training of the RPN.
        rpn_bg_iou_thresh (float): maximum IoU between the anchor and the GT box so that they can be
            considered as negative during training of the RPN.
        rpn_batch_size_per_image (int): number of anchors that are sampled during training of the RPN
            for computing the loss
        rpn_positive_fraction (float): proportion of positive anchors in a mini-batch during training
            of the RPN
        box_roi_pool (MultiScaleRoIAlign): the module which crops and resizes the feature maps in
            the locations indicated by the bounding boxes
        box_head (nn.Module): module that takes the cropped feature maps as input
        box_predictor (nn.Module): module that takes the output of box_head and returns the
            classification logits and box regression deltas.
        box_score_thresh (float): during inference, only return proposals with a classification score
            greater than box_score_thresh
        box_nms_thresh (float): NMS threshold for the prediction head. Used during inference
        box_detections_per_img (int): maximum number of detections per image, for all classes.
        box_fg_iou_thresh (float): minimum IoU between the proposals and the GT box so that they can be
            considered as positive during training of the classification head
        box_bg_iou_thresh (float): maximum IoU between the proposals and the GT box so that they can be
            considered as negative during training of the classification head
        box_batch_size_per_image (int): number of proposals that are sampled during training of the
            classification head
        box_positive_fraction (float): proportion of positive proposals in a mini-batch during training
            of the classification head
        bbox_reg_weights (Tuple[float, float, float, float]): weights for the encoding/decoding of the
            bounding boxes
        batched_nms (bool): inference box nms done independently per class (True) or globally (False)?
        indep_classif_boxes (bool): if true, a fixed-size bbox is used around each nucleus for classification.
        classification_bbox_size (float): side of the bbox for classification.
        n_fc_classif_layers (int): no of fully connected layer for the classification head.
        fc_classif_dropout (float): dropout probability for extra fully connected classification layers
        cconvhead (ClassificationConvolutions): extra covolutional layers just for classification
        sattention_head (SelfAttentionEncoder): Self attention head to improve classification based
            on other nuclei in the same fov using a tensformer encoder model.
        ignore_label (int): default is None. Label to ignore from classification loss.
        n_testtime_augmentations (int): no oftest-time augmentations.
        proposal_augmenter (RpnProposalAugmenter): this is a function or class that
            can be called to obtain an augmented realization (eg random shift)
            of object proposals from the RPN output.

    Example::

        >>> import torch
        >>> import torchvision
        >>> from torchvision.models.detection import FasterRCNN
        >>> from torchvision.models.detection.rpn import AnchorGenerator
        >>> # load a pre-trained model for classification and return
        >>> # only the features
        >>> backbone = torchvision.models.mobilenet_v2(pretrained=True).features
        >>> # FasterRCNN needs to know the number of
        >>> # output channels in a backbone. For mobilenet_v2, it's 1280
        >>> # so we need to add it here
        >>> backbone.out_channels = 1280
        >>>
        >>> # let's make the RPN generate 5 x 3 anchors per spatial
        >>> # location, with 5 different sizes and 3 different aspect
        >>> # ratios. We have a Tuple[Tuple[int]] because each feature
        >>> # map could potentially have different sizes and
        >>> # aspect ratios
        >>> anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512),),
        >>>                                    aspect_ratios=((0.5, 1.0, 2.0),))
        >>>
        >>> # let's define what are the feature maps that we will
        >>> # use to perform the region of interest cropping, as well as
        >>> # the size of the crop after rescaling.
        >>> # if your backbone returns a Tensor, featmap_names is expected to
        >>> # be [0]. More generally, the backbone should return an
        >>> # OrderedDict[Tensor], and in featmap_names you can choose which
        >>> # feature maps to use.
        >>> roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=[0],
        >>>                                                 output_size=7,
        >>>                                                 sampling_ratio=2)
        >>>
        >>> # put the pieces together inside a FasterRCNN model
        >>> model = FasterRCNN(backbone,
        >>>                    num_classes=2,
        >>>                    rpn_anchor_generator=anchor_generator,
        >>>                    box_roi_pool=roi_pooler)
        >>> model.eval()
        >>> x = [torch.rand(3, 300, 400), torch.rand(3, 500, 400)]
        >>> predictions = model(x)
    """

    def __init__(self, backbone, num_classes=None,
                 # transform parameters
                 scale_factor=2.5, scale_factor_jitter=0.25,
                 image_mean=None, image_std=None,
                 # RPN parameters
                 rpn_anchor_generator=None, rpn_head=None,
                 rpn_pre_nms_top_n_train=2000, rpn_pre_nms_top_n_test=1000,
                 rpn_post_nms_top_n_train=2000, rpn_post_nms_top_n_test=1000,
                 rpn_nms_thresh=0.7,
                 rpn_fg_iou_thresh=0.7, rpn_bg_iou_thresh=0.3,
                 rpn_batch_size_per_image=256, rpn_positive_fraction=0.5,
                 # Box parameters
                 box_roi_pool=None, box_head=None, box_predictor=None,
                 box_score_thresh=0.05, box_nms_thresh=0.5, box_detections_per_img=100,
                 box_fg_iou_thresh=0.5, box_bg_iou_thresh=0.5,
                 box_batch_size_per_image=512, box_positive_fraction=0.25,
                 bbox_reg_weights=None,
                 # added by Mohamed
                 batched_nms=True,
                 indep_classif_boxes=False,
                 classification_bbox_size=None,
                 n_fc_classif_layers=1,
                 fc_classif_dropout=0.1,
                 cconvhead=None,
                 sattention_head=None,
                 ignore_label: int = None,
                 proposal_augmenter=None,
                 n_testtime_augmentations=0
                 ):

        if not hasattr(backbone, "out_channels"):
            raise ValueError(
                "backbone should contain an attribute out_channels "
                "specifying the number of output channels (assumed to be the "
                "same for all the levels)")

        assert isinstance(rpn_anchor_generator, (AnchorGenerator, type(None)))
        assert isinstance(box_roi_pool, (MultiScaleRoIAlign, type(None)))

        if num_classes is not None:
            if box_predictor is not None:
                raise ValueError("num_classes should be None when box_predictor is specified")
        else:
            if box_predictor is None:
                raise ValueError("num_classes should not be None when box_predictor "
                                 "is not specified")

        out_channels = backbone.out_channels

        if rpn_anchor_generator is None:
            anchor_sizes = ((32,), (64,), (128,), (256,), (512,))
            aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
            rpn_anchor_generator = AnchorGenerator(
                anchor_sizes, aspect_ratios
            )
        if rpn_head is None:
            rpn_head = RPNHead(
                out_channels, rpn_anchor_generator.num_anchors_per_location()[0]
            )

        rpn_pre_nms_top_n = dict(training=rpn_pre_nms_top_n_train, testing=rpn_pre_nms_top_n_test)
        rpn_post_nms_top_n = dict(training=rpn_post_nms_top_n_train, testing=rpn_post_nms_top_n_test)

        rpn = RegionProposalNetwork(
            rpn_anchor_generator, rpn_head,
            rpn_fg_iou_thresh, rpn_bg_iou_thresh,
            rpn_batch_size_per_image, rpn_positive_fraction,
            rpn_pre_nms_top_n, rpn_post_nms_top_n, rpn_nms_thresh)

        if box_roi_pool is None:
            box_roi_pool = MultiScaleRoIAlign(
                featmap_names=[0, 1, 2, 3],
                output_size=7,
                sampling_ratio=2)

        if box_head is None:
            resolution = box_roi_pool.output_size[0]
            representation_size = 1024
            box_head = TwoMLPHead(
                out_channels * resolution ** 2,
                representation_size)

        if box_predictor is None:
            representation_size = 1024
            box_predictor = FastRCNNPredictor(
                in_channels=representation_size,
                num_classes=num_classes,
                n_fc_classif_layers=n_fc_classif_layers,
                dropout=fc_classif_dropout,
                batched_nms=batched_nms,
            )

        roi_heads = RoIHeads(
            # Box
            box_roi_pool=box_roi_pool,
            box_head=box_head,
            box_predictor=box_predictor,
            fg_iou_thresh=box_fg_iou_thresh,
            bg_iou_thresh=box_bg_iou_thresh,
            batch_size_per_image=box_batch_size_per_image,
            positive_fraction=box_positive_fraction,
            bbox_reg_weights=bbox_reg_weights,
            score_thresh=box_score_thresh,
            nms_thresh=box_nms_thresh,
            detections_per_img=box_detections_per_img,
            # added by Mohamed
            batched_nms=batched_nms,
            indep_classif_boxes=indep_classif_boxes,
            classification_bbox_size=classification_bbox_size,
            cconvhead=cconvhead,
            sattention_head=sattention_head,
            ignore_label=ignore_label,
        )

        if image_mean is None:
            image_mean = [0.485, 0.456, 0.406]
        if image_std is None:
            image_std = [0.229, 0.224, 0.225]
        # Mohamed: I changed GeneralizedRCNNTransform to take a scale factor
        #  as opposed to a fixed size to allow free size images in inference
        transform = GeneralizedRCNNTransform(
            scale_factor=scale_factor, scale_factor_jitter=scale_factor_jitter,
            image_mean=image_mean, image_std=image_std)

        super(FasterRCNN, self).__init__(
            backbone=backbone,
            rpn=rpn,
            roi_heads=roi_heads,
            transform=transform,
            # Mohamed: added this
            proposal_augmenter=proposal_augmenter,
            n_testtime_augmentations=n_testtime_augmentations,
        )


# noinspection LongLine
class TwoMLPHead(nn.Module):
    """
    Standard heads for FPN-based models

    Arguments:
        in_channels (int): number of input channels
        representation_size (int): size of the intermediate representation
    """

    def __init__(self, in_channels, representation_size):
        super(TwoMLPHead, self).__init__()

        self.fc6 = nn.Linear(in_channels, representation_size)
        self.fc7 = nn.Linear(representation_size, representation_size)

    def forward(self, x):
        x = x.flatten(start_dim=1)

        x = F.relu(self.fc6(x))
        x = F.relu(self.fc7(x))

        return x


# noinspection LongLine
class ClassificationConvolutions(nn.Sequential):
    def __init__(self, in_channels, layers, dilation=1):
        """
        Extra convolutional layes JUST for object classification using
        box pooled features. Modified from MaskRCNNHeads.
        Arguments:
            in_channels (int): number of input channels
            layers (list): feature dimensions of each convolutional layer
            dilation (int): dilation rate of kernel. The padding is set to
              the same value to preserve dimensions.
        """
        d = OrderedDict()
        next_feature = in_channels
        for layer_idx, layer_features in enumerate(layers, 1):
            d["classif_cn{}".format(layer_idx)] = nn.Conv2d(
                next_feature, layer_features, kernel_size=3,
                stride=1, padding=dilation, dilation=dilation)
            d["classif_relu{}".format(layer_idx)] = nn.ReLU(inplace=True)
            next_feature = layer_features

        super(ClassificationConvolutions, self).__init__(d)

        for name, param in self.named_parameters():
            if "weight" in name:
                nn.init.kaiming_normal_(param, mode="fan_out", nonlinearity="relu")
            # elif "bias" in name:
            #     nn.init.constant_(param, 0)


# noinspection LongLine
class SelfAttentionEncoder(nn.Module):
    """
    Self attention head to improve classification based on other nuclei
    in the same FOV. Each nucleus representation becomes influenced by (i.e.
    attentive to) other nuclei in the same FOV. The degree of attention to
    pay to self or others is fully learned. See
    "Attention is all you need", https://arxiv.org/pdf/1706.03762.pdf

    Arguments:
        representation_size (int): number of input channels
    """

    def __init__(
            self, representation_size, n_heads=8, n_layers=1,
            dim_feedforward=2048, dropout=0.1):
        super(SelfAttentionEncoder, self).__init__()

        # Transformer encoder. The result is the same size as input, but each
        # nucleus representation is aware of (i.e altered by) other nuclei
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=representation_size, nhead=n_heads,
            dim_feedforward=dim_feedforward, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=n_layers)

    def forward(self, x, fov_start_idxs: List[int]):
        """The following makes sure nuclei within the SAME FOV pay attention
        to each other, even if there are many FOVs per batch

        IMPORTANT NOTE!! The transformer model expects the input size to
          to be [S, N, E], where
          - S is the number of nuclei/proposals in FOV (i.e. no of "words"),
          - N is the number of FOVs (i.e. batch size / no. of "sentences"),
          - E is the nucleus representation ("embedding") size.

        See nn.transformer documentation as well as:
          https://discuss.pytorch.org/t/nn-transformer-explaination/53175/6

        """
        fov_end_idxs = fov_start_idxs[1:] + [int(x.shape[0])]
        batched_x = torch.cat([
            x[start:end, :][:, None, :]
            for start, end in zip(fov_start_idxs, fov_end_idxs)
        ], dim=1)
        batched_x = self.transformer_encoder(batched_x)
        batched_x = [j[:, 0, :] for j in batched_x.split(split_size=1, dim=1)]
        return torch.cat(batched_x, dim=0)


# noinspection LongLine
class FastRCNNPredictor(nn.Module):
    """
    Standard classification + bounding box regression layers
    for Fast R-CNN.

    Arguments:
        in_channels (int): number of input channels
        num_classes (int): number of output classes (including background)
    """

    def __init__(
            self, in_channels, num_classes, n_fc_classif_layers=1,
            dropout=0.1, batched_nms=True):
        super(FastRCNNPredictor, self).__init__()
        self.n_fc_classif_layers = n_fc_classif_layers
        self.batched_nms = batched_nms
        self.fc_classif_layers = {
            i: nn.Linear(in_channels, in_channels)
            for i in range(n_fc_classif_layers - 1)
        }
        self.dropout = nn.Dropout(dropout)
        self.cls_score = nn.Linear(in_channels, num_classes)  # last layer
        if self.batched_nms:
            self.bbox_pred = nn.Linear(in_channels, num_classes * 4)
        else:
            self.bbox_pred = nn.Linear(in_channels, 2 * 4)  # bkgrnd, frgrnd

    def forward(self, x, get_scores=True, get_deltas=True):
        if x.ndimension() == 4:
            assert list(x.shape[2:]) == [1, 1]
        x = x.flatten(start_dim=1)

        # classification
        if get_scores:
            scores = x + 0.
            for lno, layer in self.fc_classif_layers.items():
                scores = F.relu(layer(scores))
                scores = self.dropout(scores)
            # final layer is a simple linear transform
            scores = self.cls_score(scores)
        else:
            scores = None

        # bounding box adjustment deltas (linear transfrom)
        bbox_deltas = self.bbox_pred(x) if get_deltas else None

        return scores, bbox_deltas


model_urls = {
    'fasterrcnn_resnet50_fpn_coco':
        'https://download.pytorch.org/models/fasterrcnn_resnet50_fpn_coco-258fb6c6.pth',  # noqa
}


# noinspection LongLine
def fasterrcnn_resnet50_fpn(pretrained=False, progress=True,
                            num_classes=91, pretrained_backbone=True, **kwargs):
    """
    Constructs a Faster R-CNN model with a ResNet-50-FPN backbone.

    The input to the model is expected to be a list of tensors, each of shape ``[C, H, W]``, one for each
    image, and should be in ``0-1`` range. Different images can have different sizes.

    The behavior of the model changes depending if it is in training or evaluation mode.

    During training, the model expects both the input tensors, as well as a targets (list of dictionary),
    containing:
        - boxes (``FloatTensor[N, 4]``): the ground-truth boxes in ``[x1, y1, x2, y2]`` format, with values
          between ``0`` and ``H`` and ``0`` and ``W``
        - labels (``Int64Tensor[N]``): the class label for each ground-truth box

    The model returns a ``Dict[Tensor]`` during training, containing the classification and regression
    losses for both the RPN and the R-CNN.

    During inference, the model requires only the input tensors, and returns the post-processed
    predictions as a ``List[Dict[Tensor]]``, one for each input image. The fields of the ``Dict`` are as
    follows:
        - boxes (``FloatTensor[N, 4]``): the predicted boxes in ``[x1, y1, x2, y2]`` format, with values between
          ``0`` and ``H`` and ``0`` and ``W``
        - labels (``Int64Tensor[N]``): the predicted labels for each image
        - scores (``Tensor[N]``): the scores or each prediction

    Example::

        >>> model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)  # noqa
        >>> model.eval()
        >>> x = [torch.rand(3, 300, 400), torch.rand(3, 500, 400)]
        >>> predictions = model(x)  # noqa

    Arguments:
        pretrained (bool): If True, returns a model pre-trained on COCO train2017
        progress (bool): If True, displays a progress bar of the download to stderr

    Parameters
    ----------
    pretrained
    progress
    pretrained_backbone
    num_classes
    """
    if pretrained:
        # no need to download the backbone if pretrained is set
        pretrained_backbone = False
    backbone = resnet_fpn_backbone('resnet50', pretrained_backbone)
    model = FasterRCNN(backbone, num_classes, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls['fasterrcnn_resnet50_fpn_coco'],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model
