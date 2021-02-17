import sys
from os.path import join as opj
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.ops import MultiScaleRoIAlign
from copy import deepcopy

BASEPATH = "/home/mtageld/Desktop/cTME/"
sys.path.insert(0, BASEPATH)
from GeneralUtils import reverse_dict  # noqa
from configs.nucleus_style_defaults import NucleusCategories  # noqa
from nucls_model.DataLoadingUtils import save_cv_train_test_splits, \
    get_transform  # noqa
from nucls_model.torchvision_detection_utils.utils import collate_fn  # noqa
from nucls_model.FeatureExtractor import FeatureExtractor  # noqa
from nucls_model.FasterRCNN import ClassificationConvolutions, \
    SelfAttentionEncoder  # noqa
from nucls_model.MaskRCNN import MaskRCNNHeads  # noqa


class CoreSetQC(object):
    """paths and configs related to the QCd core set."""

    dname = 'v4_2020-04-05_FINAL_CORE'
    dataset_root = opj(BASEPATH, f'data/tcga-nucleus/{dname}/CORE_SET/QC/')
    dbpath = opj(BASEPATH, f'data/tcga-nucleus/{dname}/{dname}.sqlite')
    dataset_name = dname + '_QC'

    train_test_splits_path = opj(dataset_root, 'train_test_splits/')

    # # split to training and testing
    # save_cv_train_test_splits(
    #     root=dataset_root, savepath=train_test_splits_path,
    #     n_folds=5, by_hospital=True)


class CoreSetNoQC(object):
    """paths and configs related to the Non-QCd core set."""

    dname = CoreSetQC.dname
    dataset_root = opj(BASEPATH, f'data/tcga-nucleus/{dname}/CORE_SET/noQC/')
    dbpath = opj(BASEPATH, f'data/tcga-nucleus/{dname}/{dname}.sqlite')
    dataset_name = dname + '_noQC'


class RawEvalSets(object):
    """paths and configs related to the multirater sets as-is"""

    dname = 'CURATED_v1_2020-03-29_EVAL'
    dbpath = opj(BASEPATH, f'data/tcga-nucleus/{dname}/{dname}.sqlite')
    dataset_roots = {}
    for ev in ['E', 'B-control', 'U-control']:
        dataset_roots[ev] = opj(BASEPATH, f'data/tcga-nucleus/{dname}', ev)



class EvalSets(object):
    """paths and configs related to the parsed multirater sets (anchors)."""

    dbbase = opj(
        BASEPATH, 'results/interrater/tcga-nucleus/',
        'CURATED_v1_2020-03-29_EVAL/i1_anchors/DATASET/',
    )

    dataset_roots = {}
    dbpaths = {}
    for ev in ['E', 'U-control']:
        dataset_roots[ev] = {}
        dbpaths[ev] = {}
        for wit in ['Ps', 'NPs']:
            dataset_roots[ev][wit] = opj(dbbase, f'{wit}AreTruth_{ev}')
            dbpaths[ev][wit] = opj(
                dbbase, f'RcnnAnchors_{ev}_{wit}_AreTruth.sqlite')


class BaseDatasetConfigs(object):
    """Configs for the Nucleus dataset and loader."""

    # Dataset configs
    common_dataset = {
        'load_once': True,  # should be True to reduce disk IO
        'do_classification': True,  # FIXME: config
        'use_supercategs': False,  # 7 nucleus classes
        # 'use_supercategs': True,  # 4 nucleus classes
    }
    train_dataset = deepcopy(common_dataset)
    train_dataset.update({
        'crop_to_fov': True,
        # 'crop_size': None,  # FIXME: prototyping
        'crop_size': 300,
        # 'transforms': get_transform(tlist=None),  # FIXME: prototyping
        'transforms': get_transform(
            tlist=['hflip', 'augment_stain'],
            flip_prob=0.5,
            augment_stain_sigma1=0.5, augment_stain_sigma2=0.5,
        ),
    })
    # NOTE: these are for fasterrcnn, and are over-ridden by maskrcnn below!!
    test_dataset = deepcopy(common_dataset)
    test_dataset.update({
        # 'crop_to_fov': False,  # mFIXME: this should always be True.
        'crop_to_fov': True,  # for comparison to maskrcnn
        'crop_size': None,
        'transforms': get_transform(tlist=None),
    })

    # Dataset loader configs
    # See this nice video on setting the num_worker param:
    # https://youtu.be/kWVgvsejXsE
    # Basically, you want it to be > 0 so that at least one process
    # loads the data batch queue IN ADVANCE so that data loading is not
    # a bottleneck. Note that setting too many workers doesn't necessarily help
    # since you will already have enough batches in the queue for this to not
    # be a bottleneck. In fact, it might slow things down by increasing the
    # OVERALL CPU memory consumption and workload, especially if the batch size
    # is large!
    # For fasterrcnn, Alexnet, batch=8, 4 GPUs, sweet spot is num_workers=2
    # For fasterrcnn, Resnet18, batch=6, 4 GPUs, sweet spot is num_workers=2
    common_loader = {
        # 'num_workers': 1,  # sometimes (eg debug mode) only using 1 works
        'num_workers': 2,
        'collate_fn': collate_fn,
    }
    train_loader = deepcopy(common_loader)
    train_loader.update({
        # This is the SUB-BATCH size; "effective" batch size is a multiple
        'batch_size': 2,  # FIXME: config
        'shuffle': True,
    })
    test_loader = deepcopy(common_loader)
    test_loader.update({
        'batch_size': 1,  # FIXME: config
        'shuffle': False,
    })


class FasterRCNNConfigs(object):
    """Faster-RCNN model params."""

    # main backbone parameters
    feature_extractor_params = {
        # 'nntype': 'alexnet',  # FIXME: prototyping
        'nntype': 'resnet18',
        # 'nntype': 'resnet34',
        # 'nntype': 'resnet50',
        # 'nntype': 'resnet101',
        'pretrained': True,  # True
    }

    # num_classes includes background (i.e. real classes + 1)
    if not BaseDatasetConfigs.common_dataset['do_classification']:
        num_classes = len(NucleusCategories.puredet_categs) + 1
    elif BaseDatasetConfigs.common_dataset['use_supercategs']:
        num_classes = len(NucleusCategories.super_categs) + 1
    else:
        num_classes = len(NucleusCategories.main_categs) + 1
    ignore_label = num_classes - 1  # -> NucleusCategories.ambiguous_categs

    # Resizing parameters
    transform_parameters = {
        'scale_factor': 1200 / 300,
        # 'scale_factor': 800 / 300,
        # 'scale_factor': 1.,

        'scale_factor_jitter': 0.4,
        # 'scale_factor_jitter': 0.26,
        # 'scale_factor_jitter': 0.1,
        # 'scale_factor_jitter': None,  # no scale augmentation
    }

    # RPN parameters
    base_anchor_size = [12, 24, 48]
    anchor_sizes = []
    for j in base_anchor_size:
        anchor_sizes.append(int(j * transform_parameters['scale_factor']))
    anchor_generator_params = {
        'sizes': (tuple(anchor_sizes),),
        'aspect_ratios': ((0.5, 1.0, 2.0),),
    }
    rpn_parameters = {
        # FIXME: relevant rpn params

        # Train: We'll 3x default because we have waaay more objects
        'rpn_pre_nms_top_n_train': 6000,  # Default: 2000, Ram: 24000
        'rpn_post_nms_top_n_train': 3000,  # Default: 1000, Ram: 4000

        # TODO: when doing whole-slide inference, ron pre- and post- nms
        #  should be a function of the size of tile being fed to model
        # Test: We'll 3x default because we have waaay more objects
        'rpn_pre_nms_top_n_test': 6000,  # Default: 2000, Ram: 24000
        'rpn_post_nms_top_n_test': 3000,  # Default: 1000, Ram: 4000

        'rpn_nms_thresh': 0.7,  # Default: 0.7, Ram: 0.7
        'rpn_fg_iou_thresh': 0.7,  # Default: 0.7, Ram: 0.7
        'rpn_bg_iou_thresh': 0.3,  # Default: 0.3, Ram: 0.3
        'rpn_batch_size_per_image': 256,  # Default: 256, Ram: 256
        'rpn_positive_fraction': 0.5,  # Default: 0.5, Ram: 0.5
    }

    # Box parameters
    bbox_roialign_params = {
        'output_size': 14,  # default 7, but use 14 fpr comparison w/ maskrcnn
        # 'sampling_ratio': 2,  # FIXME: config
        'sampling_ratio': -1,  # adaptive ROIAlign sampling
    }
    box_parameters = {
        'box_score_thresh': 0.05,  # Default: 0.05, Ram:

        # FIXME: relevant box params
        # TODO: when doing whole-slide inference, box_detections_per_img
        #  should be a function of the size of tile being fed to model
        'box_nms_thresh': 0.2,  # Default: 0.5, Ram: 0.3
        'box_detections_per_img': 300,  # Default: 100, Ram: 800

        'box_fg_iou_thresh': 0.5,  # Default: 0.5, Ram: 0.5
        'box_bg_iou_thresh': 0.5,  # Default: 0.5, Ram: 0.5
        'box_batch_size_per_image': 512,  # Default: 512, Ram: 256
        'box_positive_fraction': 0.25,  # Default: 0.25, Ram: 0.25
    }

    # *- Main dict to collect params -*
    fastercnn_params = {
        # load a pre-trained model for classification and return
        # only the features. This is the network trunk
        'backbone': FeatureExtractor(**feature_extractor_params),
        'num_classes': num_classes,
        'ignore_label': ignore_label,

        # let's make the RPN generate, say, 5 x 3 anchors per spatial
        # location, with 5 different sizes and 3 different aspect
        # ratios. Each feature map has different sizes and aspect ratios
        'rpn_anchor_generator': AnchorGenerator(**anchor_generator_params),

        # let's define what are the feature maps that we will
        # use to perform the region of interest cropping, as well as
        # the size of the crop after rescaling.
        'box_roi_pool': MultiScaleRoIAlign(
            featmap_names=['0'], **bbox_roialign_params),
    }

    # FIXME! --** Additional configs to improve classification!! **--
    mohamed_customizations = {
        # Inference nms independently per class?
        'batched_nms': False,  # Default: True, False works well

        # Maybe a fixed-size bbox just for classification?
        'indep_classif_boxes': False,  # default: False
        # 'indep_classif_boxes': True,
        # 'classification_bbox_size': 64,

        # WARNING!! Extra fc layers cause HEAVY additional memory footprint!
        # Control no of fully connected classification layers
        'n_fc_classif_layers': 1,  # default: 1
        # 'n_fc_classif_layers': 2,
        # 'fc_classif_dropout': 0.1,

        # Additional convolutional layers just for classification
        # 'cconvhead': None,  # default: None
        'cconvhead': ClassificationConvolutions(
            in_channels=fastercnn_params['backbone'].out_channels,
            layers=4 * [fastercnn_params['backbone'].out_channels],
        ),

        # Self-attention head to improve classification by being attentive
        # to other nucli within the same FOV
        'sattention_head': None,  # default: None
        # 'sattention_head': SelfAttentionEncoder(
        #     representation_size=1024,  # = representation_size of box_head
        #     n_heads=8,
        #     n_layers=1,
        #     dim_feedforward=1024,
        #     dropout=0.1,
        # ),

        'proposal_augmenter': None,  # default: None
        # 'proposal_augmenter': RpnProposalAugmenter(
        #     # ops=['shift', 'resize', 'aspect'],
        #     ops=['shift'],
        #     max_shift=13 * transform_parameters['scale_factor'],
        #     # min_resize_factor=0.8,
        #     # max_resize_factor=1.2,
        # ),
    }
    if 'classification_bbox_size' in mohamed_customizations:
        mohamed_customizations['classification_bbox_size'] *= \
            transform_parameters['scale_factor']

    # Pack all prams for Faster-Rcnn
    fastercnn_params.update(transform_parameters)
    fastercnn_params.update(rpn_parameters)
    fastercnn_params.update(box_parameters)
    fastercnn_params.update(mohamed_customizations)

    # params for training the model
    training_params = {
        'n_gradient_updates': 16000,  # maskrcnn paper: 160k grad. updates
        'freeze_det_after': 15000,
        'effective_batch_size': 4,

        # monitoring params
        'print_freq': 12,  # meaningful if >= effective_batch_size / batch_size
        'window_size': None,
        'smoothing_window': 25,
        'test_evaluate_freq': 20,
        'test_maxDets': [1, 100] + [box_parameters['box_detections_per_img']],

        'n_testtime_augmentations': [0],
        # 'n_testtime_augmentations': [0, 7],  # FIXME: config

        # This is overridden by maskrcnn below (set to False)
        # Crop to fov so that the model looks at a wider field than it
        # does inference? This is also important since the dataset I made
        # deliberately looks beyond FOV to include full extent of nuclei that
        # spill over the FOV edge
        # 'crop_inference_to_fov': True, # True if model sees beyond fov
        'crop_inference_to_fov': False,  # for comparison with maskrcnn

        'optimizer_type': 'SGD',
        'optimizer_params': {
            'lr': 2e-3,  # Ram: 3e-4, Mrcnn paper: 0.02
            'momentum': 0.9,  # Ram: 0.9, Mrcnn paper: 0.9
            'weight_decay': 5e-4,  # Ram: 5e-4, Mrcnn paper: 1e-4
        },
        # 'optimizer_type': 'Adam',
        # 'optimizer_params':  {
        #     'lr': 2e-3,  # 3e-4
        #     'weight_decay': 5e-4,  # 0.0005
        # },

        'lr_scheduler_type': None,
        # 'lr_scheduler_type': 'step',
        # 'lr_scheduler_params': {
        #     'step_size': 20000,  # maskrcnn paper: 120k grad. updates
        #     'gamma': 0.5,  # maskrcnn paper: 0.1 (10x decrease)
        # },

        # 'loss_weights': None,
        'loss_weights': {
            # Even if no classification, loss_classifier should not be 0 since
            # there's an extra channel for objectness
            'loss_classifier': 1.,
            'loss_box_reg': 1.,
            'loss_objectness': 1.,
            'loss_rpn_box_reg': 1.,
        },
    }
    if training_params['crop_inference_to_fov']:
        assert not BaseDatasetConfigs.test_dataset['crop_to_fov']
        assert BaseDatasetConfigs.test_dataset['crop_size'] is None

    # weigh fovs to favor those with denser & uncommon nuclei?
    handle_class_imbalance = True
    sample_with_replacement = True


class MaskDatasetConfigs(BaseDatasetConfigs):
    train_dataset = deepcopy(BaseDatasetConfigs.train_dataset)
    test_dataset = deepcopy(BaseDatasetConfigs.test_dataset)

    # The cropper does not support sparse masks (i.e. predictions)
    # so, since the predictions cannot be cropped to the FOV, it does
    # would be unfair (i.e. give a falsely low accuracy) to not crop
    # the testing set to FOV.
    test_dataset.update({'crop_to_fov': True, 'crop_size': None})


class MaskRCNNConfigs(FasterRCNNConfigs):
    maskrcnn_params = deepcopy(FasterRCNNConfigs.fastercnn_params)

    # mask_roialign_params = {
    #     'output_size': 14,  # default: 14
    #     'sampling_ratio': 2,  # default: 2
    # }
    mask_roialign_params = deepcopy(FasterRCNNConfigs.bbox_roialign_params)

    mask_head_params = {
        'in_channels': maskrcnn_params['backbone'].out_channels,
        'layers': (256, 256, 256, 256),  # default: (256, 256, 256, 256)
        'dilation': 1,  # default: 1
    }
    maskrcnn_params.update({
        'densify_mask': False,  # FIXME: config
        'mask_roi_pool': MultiScaleRoIAlign(
            featmap_names=['0'], **mask_roialign_params),
        'mask_head': MaskRCNNHeads(**mask_head_params),
        'mask_predictor': None,
    })

    training_params = deepcopy(FasterRCNNConfigs.training_params)
    training_params['loss_weights'].update({
        'loss_mask': 1.,
    })
    # The cropper does not support sparse masks (i.e. predictions)
    # so, since the predictions cannot be cropped to the FOV, it does
    # would be unfair (i.e. give a falsely low accuracy) to not crop
    # the testing set dala loader to FOV.
    training_params.update({'crop_inference_to_fov': False})


class VisConfigs(object):

    # From Ram's paper -->
    # " green boxes indicate true-positive detections,
    #   red boxes false negatives missed by the detector,
    #   blue boxes false positives ..."
    # we use a similar idea here (not necessarily same colors)
    MATCHING_COLORS = {
        'TP': [190, 255, 35],
        'FP': [255, 255, 35],
        'FN': [255, 64, 100],
    }

    CATEG_COLORS = {
        # main categs
        'tumor_nonMitotic': [255, 0, 0],
        'tumor_mitotic': [255, 191, 0],
        'nonTILnonMQ_stromal': [0, 230, 77],
        'macrophage': [51, 102, 153],
        'lymphocyte': [0, 0, 255],
        'plasma_cell': [0, 255, 255],
        'other_nucleus': [0, 0, 0],
        'AMBIGUOUS': [80, 80, 80],

        # super categs
        'tumor_any': [255, 0, 0],
        'nonTIL_stromal': [0, 230, 77],
        'sTIL': [0, 0, 255],

        # pure detection
        'nucleus': [255, 255, 0],
    }


class PaperTablesConfigs(object):

    ncg = NucleusCategories

    base_accuracy_cols = {
        'Testing Set': 'Testing Set',
        'fold': 'Fold',
    }

    detection_accuracy_cols = {
        'n_true_nuclei_excl_ambiguous': 'N',
        'objectness AP @ 0.5': 'AP @.5',
        'objectness mAP @ 0.50:0.95': 'mAP @.5:.95',
    }

    segm_accuracy_cols = {
        'seg_n': 'N',
        'seg_medIOU': 'Median IOU',
        'seg_medDICE': 'Median DICE',
    }

    classif_accuracy_cols = {
        'n_matched_for_classif': 'N',
        'Super-classes': 'Super-classes',
        'accuracy': 'Accuracy',
        'mcc': 'MCC',
        'auroc_micro': 'AUROC (micro)',
        'auroc_macro': 'AUROC (macro)',
    }

    sclassif_accuracy_cols = {
        'Super-classes': 'Super-classes',
        'superCateg_accuracy': 'Accuracy',
        'superCateg_mcc': 'MCC',
        'superCateg_auroc_micro': 'AUROC (micro)',
        'superCateg_auroc_macro': 'AUROC (macro)',
    }
    rsclassif_accuracy_cols = reverse_dict(sclassif_accuracy_cols)

    main_categ_cols = {}
    main_categ_cols.update(base_accuracy_cols)
    main_categ_cols.update(detection_accuracy_cols)
    main_categ_cols.update(segm_accuracy_cols)
    main_categ_cols.update(classif_accuracy_cols)
    rmain_categ_cols = reverse_dict(main_categ_cols)

    percent_substrs = [
        'ap @', 'accuracy', 'auroc', 'aucroc', 'mcc', 'iou', 'dice']

    scategs = {
        'tumor_any': 'Tumor',
        'nonTIL_stromal': 'Stromal',
        'sTIL': 'sTILs',
    }
    supercateg_mcc_cols = {'superCateg_mcc': 'MCC - Overall'}
    supercateg_mcc_cols.update({
        f'superCateg_mcc_{rc}': f'MCC - {c}'
        for rc, c in scategs.items()})
    supercateg_auroc_cols = {
        'superCateg_auroc_micro': 'AUROC - Micro',
        'superCateg_auroc_macro': 'AUROC - Macro',
    }
    supercateg_auroc_cols.update({
        f'superCateg_aucroc_{rc}': f'AUROC - {c}'
        for rc, c in scategs.items()})
    supercateg_acc_meta_cols = {
        'Testing Set': 'Testing Set',
        'fold': 'Fold',
        'n_matched_for_classif': 'N',
    }
    supercateg_acc_cols = {}
    supercateg_acc_cols.update(supercateg_acc_meta_cols)
    supercateg_acc_cols.update(supercateg_mcc_cols)
    supercateg_acc_cols.update(supercateg_auroc_cols)

