import sys
import os
from os.path import join as opj
import matplotlib.pylab as plt
import numpy as np
from pandas import Series, DataFrame, read_csv
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from shutil import copyfile
from histomicstk.annotations_and_masks.annotations_to_masks_handler import \
    _visualize_annotations_on_rgb
from histomicstk.annotations_and_masks.masks_to_annotations_handler import \
    get_contours_from_mask

BASEPATH = "/home/mtageld/Desktop/cTME/"
sys.path.insert(0, BASEPATH)
from ctme.GeneralUtils import maybe_mkdir  # noqa
from ctme.nucleus_model.DataLoadingUtils import NucleusDataset, \
    get_cv_fold_slides, _crop_all_to_fov, NucleusDatasetMask  # noqa
import ctme.nucleus_model.PlottingUtils as pu  # noqa
from ctme.nucleus_model.FasterRCNN import FasterRCNN  # noqa
from ctme.nucleus_model.ModelRunner import trainNucleusModel, load_ckp, \
    evaluateNucleusModel  # noqa
from ctme.configs.nucleus_model_configs import CoreSetQC, CoreSetNoQC, \
    EvalSets, VisConfigs  # noqa
import ctme.nucleus_model.torchvision_detection_utils.transforms as tvdt  # noqa
from ctme.nucleus_model.MiscUtils import map_bboxes_using_hungarian_algorithm  # noqa
from ctme.nucleus_model.DataFormattingUtils import parse_sparse_mask_for_use  # noqa
from ctme.nucleus_model.MaskRCNN import MaskRCNN  # noqa
from ctme.configs.nucleus_style_defaults import NucleusCategories as ncg  # noqa


# %%===========================================================================

# noinspection DuplicatedCode
def run_one_fasterrcnn_fold(
        fold: int, cfg, model_root: str, model_name: str, train=True,
        vis_test=True):

    # for prototyping
    if fold == 999:
        cfg.FasterRCNNConfigs.training_params.update({
            'effective_batch_size': 4,
            'smoothing_window': 1,
            'test_evaluate_freq': 1,
        })

    model_folder = opj(model_root, f'fold_{fold}')
    maybe_mkdir(model_folder)
    checkpoint_path = opj(model_folder, f'{model_name}.ckpt')

    # %% --------------------------------------------------------------
    # Init model

    model = FasterRCNN(**cfg.FasterRCNNConfigs.fastercnn_params)

    # %% --------------------------------------------------------------
    # Prep data loaders

    train_slides, test_slides = get_cv_fold_slides(
        train_test_splits_path=CoreSetQC.train_test_splits_path, fold=fold)

    # copy train/test slides with model itself just to be safe
    for tr in ('train', 'test'):
        fname = f'fold_{fold}_{tr}.csv'
        copyfile(
            opj(CoreSetQC.train_test_splits_path, fname),
            opj(model_folder, fname),
        )

    train_dataset = NucleusDataset(
        root=CoreSetQC.dataset_root, dbpath=CoreSetQC.dbpath,
        slides=train_slides, **cfg.BaseDatasetConfigs.train_dataset)

    test_dataset = NucleusDataset(
        root=CoreSetQC.dataset_root, dbpath=CoreSetQC.dbpath,
        slides=test_slides, **cfg.BaseDatasetConfigs.test_dataset)

    # handle class imbalance
    if cfg.FasterRCNNConfigs.handle_class_imbalance:
        del cfg.BaseDatasetConfigs.train_loader['shuffle']
        cfg.BaseDatasetConfigs.train_loader['sampler'] = WeightedRandomSampler(
            weights=train_dataset.fov_weights,
            num_samples=len(train_dataset.fov_weights),
            replacement=cfg.FasterRCNNConfigs.sample_with_replacement,
        )

    # %% --------------------------------------------------------------
    # Train model

    if train:
        trainNucleusModel(
            model=model, checkpoint_path=checkpoint_path,
            data_loader=DataLoader(
                dataset=train_dataset, **cfg.BaseDatasetConfigs.train_loader),
            data_loader_test=DataLoader(
                dataset=test_dataset, **cfg.BaseDatasetConfigs.test_loader),
            **cfg.FasterRCNNConfigs.training_params)

    elif os.path.exists(checkpoint_path):
        ckpt = load_ckp(checkpoint_path=checkpoint_path, model=model)
        model = ckpt['model']

    # %% --------------------------------------------------------------
    # Visualize some predictions

    n_predict = 15
    min_iou = 0.5

    maybe_mkdir(opj(model_folder, 'predictions'))

    if vis_test:
        dataset = test_dataset
    else:
        dataset = train_dataset

    # cropper = tvdt.Cropper()

    model.eval()
    model.to('cpu')

    for imno in range(n_predict):

        # pick one image from the dataset
        imgtensor, target = dataset.__getitem__(imno)
        imname = dataset.rfovids[int(target['image_id'])]

        print(f"predicting image {imno} of {n_predict}: {imname}")

        # get prediction
        with torch.no_grad():
            output = model([imgtensor.to('cpu')])
        cpu_device = torch.device("cpu")
        output = [{k: v.to(cpu_device) for k, v in t.items()} for t in output]
        output = output[0]

        # Ignore ambiguous nuclei from matching. Note that the
        #  model already filters out anything predicted as ignore_label
        #  in inference mode, so we only need to do this for gtruth
        keep = target['iscrowd'] == 0
        trg_boxes = np.int32(target['boxes'][keep])

        # get true/false positives/negatives
        output_boxes = np.int32(output['boxes'])
        _, TP, FN, FP = map_bboxes_using_hungarian_algorithm(
            bboxes1=trg_boxes, bboxes2=output_boxes, min_iou=min_iou)

        # concat relevant bounding boxes
        relevant_bboxes = np.concatenate((
            output_boxes[TP], output_boxes[FP], trg_boxes[FN],
        ), axis=0)
        match_colors = [VisConfigs.MATCHING_COLORS['TP']] * len(TP) \
            + [VisConfigs.MATCHING_COLORS['FP']] * len(FP) \
            + [VisConfigs.MATCHING_COLORS['FN']] * len(FN)

        # get rgb
        rgb = np.uint8(imgtensor * 255.).transpose(1, 2, 0)

        # visualize true bounding boxes
        nperrow = 3
        nrows = 1
        fig, ax = plt.subplots(
            nrows, nperrow, figsize=(5 * nperrow, 5.3 * nrows))

        # truth
        axis = ax[0]
        axis.imshow(rgb)
        axis.set_title('rgb', fontsize=12)

        # prediction (objectness)
        axis = ax[1]
        axis = pu.vis_bbox(
            img=rgb, bbox=relevant_bboxes, ax=axis,
            instance_colors=match_colors, linewidth=1.5,
        )
        axis.set_title('bboxes (TP/FP/FN)', fontsize=12)

        # visualize prediction (classification)
        axis = ax[2]
        output_colors = Series(
            np.int32(output['labels'])).map(dataset.rlabelcodes).map(
            VisConfigs.CATEG_COLORS)
        axis = pu.vis_bbox(
            img=rgb, bbox=output_boxes, ax=axis,
            instance_colors=output_colors.tolist(),
            linewidth=1.5,
        )
        axis.set_title('prediction (classif.)', fontsize=12)

        # plt.show()
        plt.savefig(opj(model_folder, f'predictions/{imno}_{imname}.png'))
        plt.close()

# %%===========================================================================


# noinspection DuplicatedCode
def run_one_maskrcnn_fold(
        fold: int, cfg, model_root: str, model_name: str, qcd_training=True,
        train=True, vis_test=True, n_vis=100):

    # for prototyping
    if fold == 999:
        cfg.MaskRCNNConfigs.training_params.update({
            'effective_batch_size': 4,
            'smoothing_window': 1,
            'test_evaluate_freq': 1,
        })

    model_folder = opj(model_root, f'fold_{fold}')
    maybe_mkdir(model_folder)
    checkpoint_path = opj(model_folder, f'{model_name}.ckpt')

    # %% --------------------------------------------------------------
    # Init model

    model = MaskRCNN(**cfg.MaskRCNNConfigs.maskrcnn_params)

    # %% --------------------------------------------------------------
    # Prep data loaders

    train_slides, test_slides = get_cv_fold_slides(
        train_test_splits_path=CoreSetQC.train_test_splits_path, fold=fold)

    # copy train/test slides with model itself just to be safe
    for tr in ('train', 'test'):
        fname = f'fold_{fold}_{tr}.csv'
        copyfile(
            opj(CoreSetQC.train_test_splits_path, fname),
            opj(model_folder, fname),
        )

    # training data optionally QCd
    if qcd_training:
        train_dataset = NucleusDatasetMask(
            root=CoreSetQC.dataset_root, dbpath=CoreSetQC.dbpath,
            slides=train_slides, **cfg.MaskDatasetConfigs.train_dataset)
    else:
        train_dataset = NucleusDatasetMask(
            root=CoreSetNoQC.dataset_root, dbpath=CoreSetNoQC.dbpath,
            slides=train_slides, **cfg.MaskDatasetConfigs.train_dataset)

    # test set is always the QC'd data
    test_dataset = NucleusDatasetMask(
        root=CoreSetQC.dataset_root, dbpath=CoreSetQC.dbpath,
        slides=test_slides, **cfg.MaskDatasetConfigs.test_dataset)

    # handle class imbalance
    if cfg.MaskRCNNConfigs.handle_class_imbalance:
        del cfg.BaseDatasetConfigs.train_loader['shuffle']
        cfg.BaseDatasetConfigs.train_loader['sampler'] = WeightedRandomSampler(
            weights=train_dataset.fov_weights,
            num_samples=len(train_dataset.fov_weights),
            replacement=cfg.MaskRCNNConfigs.sample_with_replacement,
        )

    # %% --------------------------------------------------------------
    # Train model

    if train:
        trainNucleusModel(
            model=model, checkpoint_path=checkpoint_path,
            data_loader=DataLoader(
                dataset=train_dataset, **cfg.MaskDatasetConfigs.train_loader),
            data_loader_test=DataLoader(
                dataset=test_dataset, **cfg.MaskDatasetConfigs.test_loader),
            **cfg.MaskRCNNConfigs.training_params)

    elif os.path.exists(checkpoint_path):
        ckpt = load_ckp(checkpoint_path=checkpoint_path, model=model)
        model = ckpt['model']

    # %% --------------------------------------------------------------
    # Visualize some predictions

    min_iou = 0.5
    vis_props = {'linewidth': 0.15, 'text': False}

    maybe_mkdir(opj(model_folder, 'predictions'))

    if vis_test:
        dataset = test_dataset
    else:
        dataset = train_dataset

    # cropper = tvdt.Cropper()

    model.eval()
    model.to('cpu')

    # for imno in range(n_vis):
    tovis = list(np.random.choice(len(dataset), size=(n_vis,)))
    for imidx, imno in enumerate(tovis):

        # pick one image from the dataset
        imgtensor, target = dataset.__getitem__(imno)
        imname = dataset.rfovids[int(target['image_id'])]

        print(f"predicting image {imidx} of {n_vis}: {imname}")

        # get prediction
        with torch.no_grad():
            output = model([imgtensor.to('cpu')])
        cpu_device = torch.device('cpu')
        output = [{k: v.to(cpu_device) for k, v in t.items()} for t in output]
        output = output[0]

        # Ignore ambiguous nuclei from matching. Note that the
        #  model already filters out anything predicted as ignore_label
        #  in inference mode, so we only need to do this for gtruth
        keep = target['iscrowd'] == 0
        trg_boxes = np.int32(target['boxes'][keep])

        # get true/false positives/negatives
        output_boxes = np.int32(output['boxes'])
        _, TP, FN, FP = map_bboxes_using_hungarian_algorithm(
            bboxes1=trg_boxes, bboxes2=output_boxes, min_iou=min_iou)

        # concat relevant bounding boxes
        relevant_bboxes = np.concatenate((
            output_boxes[TP], output_boxes[FP], trg_boxes[FN],
        ), axis=0)
        match_colors = [VisConfigs.MATCHING_COLORS['TP']] * len(TP) \
            + [VisConfigs.MATCHING_COLORS['FP']] * len(FP) \
            + [VisConfigs.MATCHING_COLORS['FN']] * len(FN)

        # just to comply with histomicstk default style
        rgtcodes = {
            k: {
                'group': v,
                'color': f'rgb(' + ','.join(str(c) for c in VisConfigs.CATEG_COLORS[v]) + ')',
            }
            for k, v in dataset.rlabelcodes.items()
        }

        # extract contours +/ condensed masks (truth)
        # noinspection PyTupleAssignmentBalance
        _, _, contoursdf_truth = parse_sparse_mask_for_use(
            sparse_mask=np.uint8(target['masks']),
            rgtcodes=rgtcodes, labels=target['labels'].tolist(),
        )

        # extract contours +/ condensed masks (prediction)
        output_labels = np.int32(output['labels'])
        output_labels = output_labels.tolist()
        if not model.transform.densify_mask:
            # output mask is sparse
            # noinspection PyTupleAssignmentBalance
            _, _, contoursdf_prediction = parse_sparse_mask_for_use(
                sparse_mask=np.uint8(output['masks'][:, 0, :, :] > 0.5),
                rgtcodes=rgtcodes, labels=output_labels,
            )
        else:
            # output mask is already dense
            contoursdf_prediction = get_contours_from_mask(
                MASK=output['masks'].numpy(),
                GTCodes_df=DataFrame.from_records(data=[
                    {
                        'group': rgtcodes[label]['group'],
                        'GT_code': idx + 1,
                        'color': rgtcodes[label]['color']
                    }
                    for idx, label in enumerate(output_labels)
                ]),
                MIN_SIZE=1,
                get_roi_contour=False,
            )

        # get rgb
        rgb = np.uint8(imgtensor * 255.).transpose(1, 2, 0)

        # visualize bounding boxes and masks
        nperrow = 4
        nrows = 1
        fig, ax = plt.subplots(nrows, nperrow,
                               figsize=(5 * nperrow, 5.3 * nrows))

        # just the image
        axis = ax[0]
        axis.imshow(rgb)
        axis.set_title('rgb', fontsize=12)

        # relevant predicted (TP, FP) & true (FN) boxes
        axis = ax[1]
        axis = pu.vis_bbox(
            img=rgb, bbox=relevant_bboxes, ax=axis,
            instance_colors=match_colors, linewidth=1.5,
        )
        axis.set_title('Bboxes detection (TP/FP/FN)', fontsize=12)

        # predicted masks
        axis = ax[2]
        prediction_vis = _visualize_annotations_on_rgb(
            rgb=rgb,
            contours_list=contoursdf_prediction.to_dict(orient='records'),
            **vis_props)
        axis.imshow(prediction_vis)
        axis.set_title('Predicted masks + classif.', fontsize=12)

        # true masks
        axis = ax[3]
        truth_vis = _visualize_annotations_on_rgb(
            rgb=rgb, contours_list=contoursdf_truth.to_dict(orient='records'),
            **vis_props)
        axis.imshow(truth_vis)
        axis.set_title('True masks/bboxes + classif.', fontsize=12)

        # plt.show()
        plt.savefig(opj(model_folder, f'predictions/{imno}_{imname}.png'))
        plt.close()

# %%===========================================================================

# noinspection DuplicatedCode
def evaluate_maskrcnn_fold_on_inferred_truth(
        fold: int, cfg, model_root: str, model_name: str,
        whoistruth='Ps', evalset='E', getmetrics=True, n_vis=53):

    model_folder = opj(model_root, f'fold_{fold}')
    checkpoint_path = opj(model_folder, f'{model_name}.ckpt')
    savepath = opj(model_folder, f'Eval_{whoistruth}AreTruth_{evalset}')
    maybe_mkdir(savepath)

    # %% --------------------------------------------------------------
    # Init model

    model = MaskRCNN(**cfg.MaskRCNNConfigs.maskrcnn_params)

    # %% --------------------------------------------------------------
    # Prep data loaders

    slides = read_csv(opj(
        model_folder, f'fold_{fold}_test.csv')).loc[:, 'slide_name'].tolist()
    dataset = NucleusDatasetMask(
        root=EvalSets.dataset_roots[evalset][whoistruth],
        dbpath=EvalSets.dbpaths[evalset][whoistruth],
        slides=slides, **cfg.MaskDatasetConfigs.test_dataset)

    # %% --------------------------------------------------------------
    # Evaluate model

    ckpt = load_ckp(checkpoint_path=checkpoint_path, model=model)
    model = ckpt['model']

    if getmetrics:
        ecfgs = {
            k: v for k, v in cfg.MaskRCNNConfigs.training_params.items() if k in [
                'test_maxDets',
                'n_testtime_augmentations',
                'crop_inference_to_fov'
            ]
        }
        tsls = evaluateNucleusModel(
            model=model, checkpoint_path=checkpoint_path,
            dloader=DataLoader(
                dataset=dataset, **cfg.MaskDatasetConfigs.test_loader),
            **ecfgs)

        # save results
        for i, tsl in enumerate(tsls):
            with open(opj(savepath, f'testingMetrics_{i}.txt'), 'w') as f:
                f.write(str(tsl)[1:-1].replace(', ', '\n'))

    # %% --------------------------------------------------------------
    # Visualize some predictions

    min_iou = 0.5
    vis_props = {'linewidth': 0.15, 'text': False}

    maybe_mkdir(opj(savepath, 'predictions'))

    # cropper = tvdt.Cropper()

    model.eval()
    model.to('cpu')

    for imno in range(min(n_vis, len(dataset))):

        # pick one image from the dataset
        imgtensor, target = dataset.__getitem__(imno)
        imname = dataset.rfovids[int(target['image_id'])]

        print(f"visualizing image {imno} of {n_vis}: {imname}")

        # get prediction
        with torch.no_grad():
            output = model([imgtensor.to('cpu')])
        cpu_device = torch.device('cpu')
        output = [{k: v.to(cpu_device) for k, v in t.items()} for t in output]
        output = output[0]

        # Ignore ambiguous nuclei from matching. Note that the
        #  model already filters out anything predicted as ignore_label
        #  in inference mode, so we only need to do this for gtruth
        keep = target['iscrowd'] == 0
        trg_boxes = np.int32(target['boxes'][keep])

        # get true/false positives/negatives
        output_boxes = np.int32(output['boxes'])
        _, TP, FN, FP = map_bboxes_using_hungarian_algorithm(
            bboxes1=trg_boxes, bboxes2=output_boxes, min_iou=min_iou)

        # concat relevant bounding boxes
        relevant_bboxes = np.concatenate((
            output_boxes[TP], output_boxes[FP], trg_boxes[FN],
        ), axis=0)
        match_colors = [VisConfigs.MATCHING_COLORS['TP']] * len(TP) \
            + [VisConfigs.MATCHING_COLORS['FP']] * len(FP) \
            + [VisConfigs.MATCHING_COLORS['FN']] * len(FN)

        # just to comply with histomicstk default style
        rgtcodes = {
            k: {
                'group': v,
                'color': f'rgb(' + ','.join(str(c) for c in VisConfigs.CATEG_COLORS[v]) + ')',
            }
            for k, v in dataset.rlabelcodes.items()
        }

        # extract contours +/ condensed masks (truth)
        # noinspection PyTupleAssignmentBalance
        _, _, contoursdf_truth = parse_sparse_mask_for_use(
            sparse_mask=np.uint8(target['masks']),
            rgtcodes=rgtcodes, labels=target['labels'].tolist(),
        )

        # extract contours +/ condensed masks (prediction)
        output_labels = np.int32(output['labels'])
        output_labels = output_labels.tolist()
        if not model.transform.densify_mask:
            # output mask is sparse
            # noinspection PyTupleAssignmentBalance
            _, _, contoursdf_prediction = parse_sparse_mask_for_use(
                sparse_mask=np.uint8(output['masks'][:, 0, :, :] > 0.5),
                rgtcodes=rgtcodes, labels=output_labels,
            )
        else:
            # output mask is already dense
            contoursdf_prediction = get_contours_from_mask(
                MASK=output['masks'].numpy(),
                GTCodes_df=DataFrame.from_records(data=[
                    {
                        'group': rgtcodes[label]['group'],
                        'GT_code': idx + 1,
                        'color': rgtcodes[label]['color']
                    }
                    for idx, label in enumerate(output_labels)
                ]),
                MIN_SIZE=1,
                get_roi_contour=False,
            )

        # get rgb
        rgb = np.uint8(imgtensor * 255.).transpose(1, 2, 0)

        # visualize bounding boxes and masks
        nperrow = 4
        nrows = 1
        fig, ax = plt.subplots(nrows, nperrow,
                               figsize=(5 * nperrow, 5.3 * nrows))

        # just the image
        axis = ax[0]
        axis.imshow(rgb)
        axis.set_title('rgb', fontsize=12)

        # relevant predicted (TP, FP) & true (FN) boxes
        axis = ax[1]
        axis = pu.vis_bbox(
            img=rgb, bbox=relevant_bboxes, ax=axis,
            instance_colors=match_colors, linewidth=1.5,
        )
        axis.set_title('Bboxes detection (TP/FP/FN)', fontsize=12)

        # predicted masks
        axis = ax[2]
        prediction_vis = _visualize_annotations_on_rgb(
            rgb=rgb,
            contours_list=contoursdf_prediction.to_dict(orient='records'),
            **vis_props)
        axis.imshow(prediction_vis)
        axis.set_title('Predicted masks + classif.', fontsize=12)

        # true masks
        axis = ax[3]
        truth_vis = _visualize_annotations_on_rgb(
            rgb=rgb, contours_list=contoursdf_truth.to_dict(orient='records'),
            **vis_props)
        axis.imshow(truth_vis)
        axis.set_title('True masks/bboxes + classif.', fontsize=12)

        # plt.show()
        plt.savefig(opj(savepath, f'predictions/{imno}_{imname}.png'))
        plt.close()

# %%===========================================================================
