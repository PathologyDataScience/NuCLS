# source: https://github.com/pytorch/vision/blob/master/references/detection/

import math
import time
import torch
import torchvision.models.detection.mask_rcnn
from sklearn.metrics import roc_auc_score, matthews_corrcoef
import numpy as np
from scipy.special import softmax

import sys

import nucls_model.torchvision_detection_utils.utils as utils  # noqa
from nucls_model.torchvision_detection_utils.coco_utils import get_coco_api_from_dataset  # noqa
from nucls_model.torchvision_detection_utils.coco_eval import CocoEvaluator  # noqa
from nucls_model.MaskRCNN import MaskRCNN  # noqa
import nucls_model.torchvision_detection_utils.transforms as tvdt  # noqa
from nucls_model.DataLoadingUtils import _crop_all_to_fov  # noqa
from nucls_model.MiscUtils import map_bboxes_using_hungarian_algorithm  # noqa


# noinspection LongLine
def train_one_epoch(
        model, device, optimizer, data_loader, effective_batch_size=None,
        epoch=1, lr_scheduler=None, loss_weights=None,
        print_freq=1, window_size=20):
    model.training()
    metric_logger = utils.MetricLogger(delimiter="  ", window_size=window_size)
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)

    if effective_batch_size is None:
        effective_batch_size = data_loader.batch_size
    else:
        assert effective_batch_size % data_loader.batch_size == 0

    # Strategy: We pass one "subbatch" through the model, then get the loss,
    # but do NOT backprop .. the loss from multiple subbatches are accumulated
    # together so the effective batch size is bigger that what can fit GPU
    subbatches_per_grup = int(effective_batch_size / data_loader.batch_size)

    global_subbatch_count = 0
    subbatch = 0
    grup_loss = 0.  # total
    grup_losses = {}  # breakdown

    for images, targets in metric_logger.log_every(data_loader, print_freq, header):

        # before I forget
        subbatch += 1
        global_subbatch_count += 1

        # move subbatch to device
        images = list(image.to(device) for image in images)
        if isinstance(targets[0], dict):
            # faster/maskrcnn
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        else:
            # pure classification (targets is a list)
            targets = list(target.to(device) for target in targets)

        # forward pass
        loss_dict = model(images, targets)

        # Multiply different loss types by a weight
        if loss_weights is None:
            loss_weights = {k: 1.0 for k in loss_dict}
        loss_dict = {
            lname: loss * loss_weights[lname]
            for lname, loss in loss_dict.items()
        }

        # reduce losses over all GPUs
        loss_dict_reduced = utils.reduce_dict(loss_dict)

        # stop if nan loss
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())
        loss_value = losses_reduced.item()
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)
            # print("One of the losses if nan! Replacing with zero for now.")
            # print(loss_dict_reduced)
            # for k, v in loss_dict_reduced.items():
            #     loss_dict_reduced[k][torch.isnan(v)] = 0.
            # losses_reduced = sum(loss for loss in loss_dict_reduced.values())
            # loss_value = losses_reduced.item()

        # Clear old gradients from the last step, we already have the loss
        # (else you’d accumulate gradients from all loss.backward() calls)
        optimizer.zero_grad()

        # update overall loss (per effective batch size)
        grup_loss += loss_value
        for k, v in loss_dict_reduced.items():
            if k in grup_losses:
                grup_losses[k] += float(v)
            else:
                grup_losses[k] = float(v)

        # Last subbatch, time to backpropagate!
        if subbatch == subbatches_per_grup:

            # total loss, averaged over subbatches in this batch
            losses = sum(loss for lname, loss in loss_dict.items())
            losses = (losses + grup_loss - loss_value) / subbatches_per_grup

            # compute the derivative of the loss w.r.t. the parameters
            # (or anything requiring gradients) using backpropagation
            losses.backward()

            # take a step based on the gradients of the parameters
            optimizer.step()

            # maybe update learning rate
            if lr_scheduler is not None:
                lr_scheduler.step()

            # get floats for logging
            grup_loss /= subbatches_per_grup
            grup_losses = {
                k: v / subbatches_per_grup for k, v in grup_losses.items()
            }

            # update logger
            metric_logger.update(loss=grup_loss, **grup_losses)
            metric_logger.update(lr=optimizer.param_groups[0]["lr"])

            # reset subbatch index and effective batch losses
            subbatch = 0
            grup_loss = 0.
            grup_losses = {k: 0. for k in grup_losses}

    return metric_logger


# noinspection LongLine
def _get_iou_types(model):
    model_without_ddp = model
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model_without_ddp = model.module
    iou_types = ["bbox"]
    # Mohamed: I edited this to refer to my MaskRCNN
    if isinstance(model_without_ddp, MaskRCNN):
        iou_types.append("segm")
    if isinstance(model_without_ddp, torchvision.models.detection.KeypointRCNN):
        iou_types.append("keypoints")
    return iou_types


def _update_classification_metrics(
        metrics_dict, all_labels, rlabelcodes,
        all_scores=None, output_labels=None,
        codemap=None, prefix='',):
    """IMPORTANT NOTE: This assumes that all_labels start at zero and
    are contiguous, and that all_scores is shaped (n_nuclei, n_classes),
    where n_classes is the REAL number of classes.
    """
    if len(all_labels) < 1:
        return metrics_dict

    pexist = all_scores is not None
    if not pexist:
        assert output_labels is not None

    # maybe group scores from classes that belong to the same supercateg
    if codemap is not None:
        tmp_lbls = all_labels.copy()
        tmp_scrs = np.zeros(all_scores.shape) if pexist else None
        for k, v in codemap.items():
            # remap labels
            tmp_lbls[all_labels == k] = v
            # aggregate probab. for classes to be grouped
            if pexist:
                tmp_scrs[:, v] += all_scores[:, k]
        all_labels = tmp_lbls
        all_scores = tmp_scrs

    unique_classes = np.unique(all_labels).tolist()
    n_classes = len(unique_classes)

    if pexist:
        all_preds = np.argmax(all_scores, 1)
    else:
        all_preds = output_labels

    if n_classes > 0:
        # accuracy
        metrics_dict[f'{prefix}accuracy'] = np.mean(0 + (all_preds == all_labels))

        # Mathiew's Correlation Coefficient
        metrics_dict[f'{prefix}mcc'] = matthews_corrcoef(y_true=all_labels, y_pred=all_preds)

        # Class confusions (unnormalized, just numbers)
        for tcc, tc in rlabelcodes.items():
            for pcc, pc in rlabelcodes.items():
                coln = f'{prefix}confusion_trueClass-{tc}_predictedClass-{pc}'
                keep1 = 0 + (all_labels == tcc)
                keep2 = 0 + (all_preds == pcc)
                metrics_dict[coln] = np.sum(0 + ((keep1 + keep2) == 2))

    if n_classes > 1:

        # Class-by-class accuracy

        trg = np.zeros((len(all_labels), n_classes))
        scr = np.zeros((len(all_labels), n_classes))

        for cid, cls in enumerate(unique_classes):

            cls_name = rlabelcodes[cls]

            # Accuracy
            tr = 0 + (all_labels == cls)
            pr = 0 + (all_preds == cls)
            metrics_dict[f'{prefix}accuracy_{cls_name}'] = np.mean(0 + (tr == pr))

            # Mathiew's Correlation Coefficient
            metrics_dict[f'{prefix}mcc_{cls_name}'] = matthews_corrcoef(y_true=tr, y_pred=pr)

            # ROC AUC. Note that it's only defined for classes present in gt
            if pexist:
                trg[:, cid] = 0 + (all_labels == cls)
                scr[:, cid] = all_scores[:, cls]
                metrics_dict[f'{prefix}aucroc_{cls_name}'] = roc_auc_score(
                    y_true=trg[:, cid], y_score=all_scores[:, cid])

        # renormalize with softmax & get rocauc
        if pexist:
            scr = softmax(scr, -1)
            metrics_dict[f'{prefix}auroc_micro'] = roc_auc_score(
                y_true=trg, y_score=scr, multi_class='ovr', average='micro')
            metrics_dict[f'{prefix}auroc_macro'] = roc_auc_score(
                y_true=trg, y_score=scr, multi_class='ovr', average='macro')

        print(f"\nClassification results: {prefix}")
        for k, v in metrics_dict.items():
            if k.startswith(prefix) and ('confusion_' not in k):
                print(f'{k}: {v}')


# noinspection PyPep8Naming,LongLine
@torch.no_grad()
def evaluate(
        model, data_loader, device, maxDets=None, crop_inference_to_fov=False):
    # See: https://cocodataset.org/#detection-eval

    # NOTE: The coco evaluator (and what's reported in FasterRCNN and
    #  maskrcnn papers) combines detection and classification by
    #  considering something to be detected only if it's from the same
    #  class. eg. If the model places a bounding box and labels it "traffic
    #  light", but in reality that location has a "person", this is
    #  considered a false positive traffic light and a false negative
    #  person. We'd like to get this metric, sure, but we're also
    #  interested in classic detection .. i.e. just "is there a nucleus?"
    #  so we get AP using both the full set of classes AS WELL AS
    #  a remapped class set where anything is considered a "nucleus"

    n_threads = torch.get_num_threads()
    # mFIXME remove this and make paste_masks_in_image run on the GPU
    torch.set_num_threads(1)
    cpu_device = torch.device("cpu")
    model.eval()

    # iou_types = _get_iou_types(model)
    iou_types = ['bbox']  # segmAP is meaningless in my hybrid bbox/segm dataset
    maxDets = [1, 10, 100] if maxDets is None else maxDets
    cropper = tvdt.Cropper() if crop_inference_to_fov else None

    # combined detection & classification precision/recall
    dst = data_loader.dataset
    coco = get_coco_api_from_dataset(dst, crop_inference_to_fov=crop_inference_to_fov)
    coco_evaluator = CocoEvaluator(coco, iou_types, maxDets=maxDets)
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    # precision/recall for just detection (objectness)
    classification = dst.do_classification
    if classification:

        # IMPORTANT: REVERSE ME AFTER DEFINING COCO API
        dst.do_classification = False
        dst.set_labelmaps()

        metric_logger_objectness = utils.MetricLogger(delimiter="  ")
        coco_objectness = get_coco_api_from_dataset(
            dst, crop_inference_to_fov=crop_inference_to_fov)
        coco_evaluator_objectness = CocoEvaluator(
            coco_objectness, iou_types, maxDets=maxDets)

        # IMPORTANT: THIS LINE IS CRITICAL
        dst.do_classification = True
        dst.set_labelmaps()

    else:
        metric_logger_objectness = None
        # noinspection PyUnusedLocal
        coco_objectness = None
        coco_evaluator_objectness = None

    n_true = 0
    n_pred = 0
    n_matched = 0
    cltargets = []
    clprobabs = []
    cloutlabs = []
    seg_intersects = []
    seg_sums = []

    def _get_categnames(prefix):
        if prefix == '':
            return dst.categs_names
        return dst.supercategs_names

    for images, targets in metric_logger.log_every(data_loader, 100, header):
        images = list(img.to(device) for img in images)
        targets = list(targets)

        # uncomment if GPU
        # torch.cuda.synchronize()

        model_time = time.time()
        outputs = model(images)
        outputs = [
            {k: v.to(cpu_device) for k, v in t.items() if v is not None}
            for t in outputs
        ]
        model_time = time.time() - model_time

        if crop_inference_to_fov:
            images, targets, outputs = _crop_all_to_fov(
                images=images, targets=targets, outputs=outputs,
                cropper=cropper)

        # combined detection & classification precision/recall
        res = {
            target["image_id"].item(): output
            for target, output in zip(targets, outputs)}
        evaluator_time = time.time()
        coco_evaluator.update(res)
        evaluator_time = time.time() - evaluator_time
        metric_logger.update(
            model_time=model_time, evaluator_time=evaluator_time)

        probabs_exist = 'probabs' in outputs[0]

        if classification:

            # IMPORTANT NOTE: The way that FasterRCNN is implemented
            # assigns each box prediction a confidence score and a label. This
            # is NOT the same as the "traditional" classifier where there is a
            # confidence score for ALL classes per object/pixel. Instead, here
            # the class logits are "flattened" so that each box-label
            # combination is considered separately, then the NMS is done
            # independently per class. Long story short, each box only has
            # one label and confidence

            # Match truth to outputs and only count matched objects for
            # classification accuracy stats
            for target, output in zip(targets, outputs):

                # Match, ignoring ambiguous nuclei. Note that the model
                #  already filters out anything predicted as ignore_label
                #  in inference mode, so we only need to do this for gtruth
                keep = target['iscrowd'] == 0
                cltrg_boxes = np.int32(target['boxes'][keep])
                cltrg_labels = np.int32(target['labels'][keep])
                keep_target, keep_output, _, _ = \
                    map_bboxes_using_hungarian_algorithm(
                        bboxes1=cltrg_boxes,
                        bboxes2=np.int32(output['boxes']),
                        min_iou=0.5)

                # classification performance
                n_true += cltrg_boxes.shape[0]
                n_pred += output['boxes'].shape[0]
                n_matched += len(keep_output)
                cltargets.extend(cltrg_labels[keep_target].tolist())
                if probabs_exist:
                    clprobabs.extend(
                        np.float32(output['probabs'])[keep_output, :].tolist()
                    )
                else:
                    cloutlabs.extend(
                        np.int32(output['labels'])[keep_output].tolist()
                    )

                # FIXME: for now, we just assess this if classification because
                #   otherwise I'll need to refactor the function output
                # segmentation performance
                if 'masks' in target:
                    ismask = np.int32(target['ismask'])[keep_target] == 1
                    tmask = np.int32(target['masks'])[keep_target, ...][ismask, ...]
                    if not model.transform.densify_mask:
                        omask = np.int32(output['masks'] > 0.5)
                        omask = omask[:, 0, :, :]
                    else:
                        omask = np.int32(output['masks'])
                        obj_ids = np.arange(1, omask.max() + 1)
                        omask = omask == obj_ids[:, None, None]
                        omask = 0 + omask
                    omask = omask[keep_output, ...][ismask, ...]
                    for i in range(tmask.shape[0]):
                        sms = tmask[i, ...].sum() + omask[i, ...].sum()
                        isc = np.sum(
                            0 + ((tmask[i, ...] + omask[i, ...]) == 2)
                        )
                        if (sms > 0) and (isc > 0):
                            seg_sums.append(sms)
                            seg_intersects.append(isc)

            # FIXME (low priority): have this use a map from the data loader
            #   labelcodes to justdetection code (eg 2 -> 1, 3 -> 1, etc)
            #   instead of hardcoding the assumption that "nucleus" will
            #   always have the code 1. Note that the model already filters
            #   out anything predicted as ignore_label.
            # remap predictions to just "nucleus". Note that the labels
            # have already been remapped during indexing of the coco API.
            # NEEDLESS TO SAY, this must happen AFTER we've assigned
            # the classifications to the classification_outputs list
            for _, output in res.items():
                output['labels'] = 1 + (0 * output['labels'])

            # precision/recall for just detection (objectness)
            evaluator_time = time.time()
            coco_evaluator_objectness.update(res)
            evaluator_time = time.time() - evaluator_time
            metric_logger_objectness.update(
                model_time=model_time, evaluator_time=evaluator_time)

    # combined detection & classification precision/recall
    # gather the stats from all processes & accumulate preds from all imgs
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    coco_evaluator.synchronize_between_processes()
    coco_evaluator.accumulate()
    coco_evaluator.summarize()

    if classification:
        # Init classification results
        classification_metrics = {
            'n_true_nuclei_excl_ambiguous': n_true,
            'n_predicted_nuclei': n_pred,
            'n_matched_for_classif': n_matched,
        }
        for prefix in ['', 'superCateg_']:
            categs_names = _get_categnames(prefix)
            classification_metrics.update({
                f'{prefix}{k}': np.nan
                for k in ['accuracy', 'auroc_micro', 'auroc_macro', 'mcc']
            })
            # Class-by-class
            classification_metrics.update({
                f'{prefix}accuracy_{cls_name}': np.nan
                for cls_name in categs_names
            })
            classification_metrics.update({
                f'{prefix}mcc_{cls_name}': np.nan
                for cls_name in categs_names
            })
            if probabs_exist:
                classification_metrics.update({
                    f'{prefix}aucroc_{cls_name}': np.nan
                    for cls_name in categs_names
                })
        for prefix in ['', 'superCateg_']:
            categs_names = _get_categnames(prefix)
            classification_metrics.update({
                f'{prefix}confusion_trueClass-{tc}_predictedClass-{pc}': 0
                for tc in categs_names
                for pc in categs_names
            })
        # segmentation -- restricted to matched nuclei with available seg
        if len(seg_sums) > 0:
            seg_intersects = np.array(seg_intersects)
            seg_sums = np.array(seg_sums)
            intersect = np.sum(seg_intersects)
            sums = np.sum(seg_sums)
            ious = seg_intersects / (seg_sums - seg_intersects)
            dices = 2. * seg_intersects / seg_sums
            classification_metrics.update({
                # overall
                'seg_intersect': intersect,
                'seg_sum': sums,
                'seg_IOU': intersect / (sums - intersect),
                'seg_DICE': 2. * intersect / sums,
                # by nucleus
                'seg_n': len(ious),
                'seg_medIOU': np.median(ious),
                'seg_medDICE': np.median(dices),
            })

        metric_logger_objectness.synchronize_between_processes()
        print("\nAveraged stats (OBJECTNESS):", metric_logger_objectness)
        coco_evaluator_objectness.synchronize_between_processes()
        coco_evaluator_objectness.accumulate()
        coco_evaluator_objectness.summarize()

        # NOTE: WE MAKE SURE ALL LABELMAPS BELOW START AT ZERO SINCE THE
        # FUNCTION _update_classification_metrics DOES AN ARGMAX INTERNALLY
        # SO FIRST COLUMN CORRESPONDS TO ZERO'TH CLASS, WHICH CORRESPONDS TO
        # LABEL = 1 IN OUR DATASET AND MODEL
        # classification accuracy without remapping
        clkwargs = {
            'metrics_dict': classification_metrics,
            'all_labels': np.array(cltargets) - 1,
            'rlabelcodes': {
                k - 1: v
                for k, v in dst.rlabelcodes.items() if v != 'AMBIGUOUS'
            },
            'codemap': None,
            'prefix': 'superCateg_' if dst.use_supercategs else '',
        }
        if probabs_exist:
            clkwargs['all_scores'] = np.array(clprobabs)
        else:
            clkwargs['output_labels'] = np.array(cloutlabs)
        _update_classification_metrics(**clkwargs)

        # FIXME (low priority): this hard-codes the name of ambiguous categ
        # classification accuracy mapped to supercategs
        if not dst.use_supercategs:
            clkwargs.update({
                'rlabelcodes': {
                    k - 1: v
                    for k, v in dst.supercategs_rlabelcodes.items()
                    if v != 'AMBIGUOUS'
                },
                'codemap': {
                    k - 1: v - 1
                    for k, v in dst.main_codes_to_supercategs_codes.items()
                    if dst.supercategs_rlabelcodes[v] != 'AMBIGUOUS'
                },
                'prefix': 'superCateg_',
            })
            _update_classification_metrics(**clkwargs)
    else:
        classification_metrics = {}

    torch.set_num_threads(n_threads)

    return coco_evaluator, coco_evaluator_objectness, classification_metrics
