import os
import torch
from torch.utils.data import DataLoader
# from torch.nn.parallel import DistributedDataParallel
from torch.nn import DataParallel
from pandas import DataFrame
import numpy as np
import pickle

from nucls_model.torchvision_detection_utils.engine import \
    train_one_epoch, evaluate
import nucls_model.PlottingUtils as pu


ISCUDA = torch.cuda.is_available()
if ISCUDA:
    try:
        NGPUS = len(os.environ["CUDA_VISIBLE_DEVICES"].split(','))
    except KeyError:  # gpus available but not assigned
        NGPUS = 4
else:
    NGPUS = 0


def running_mean(x, N):
    # source: https://stackoverflow.com/questions/13728392/ ...
    # moving-average-or-running-mean/27681394#27681394
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[N:] - cumsum[:-N]) / float(N)


def load_ckp(checkpoint_path, model, optimizer=None):
    """
    Source https://towardsdatascience.com/how-to-save-and-load-a-model-in- ...
    ... pytorch-with-a-complete-example-c2920e617dee

    checkpoint_path: path to save checkpoint
    model: model that we want to load checkpoint parameters into
    optimizer: optimizer we defined in previous training
    """
    if torch.cuda.is_available():
        extra = {}
    else:
        extra = {
            'map_location': lambda storage, loc: storage,
            # 'map_location': {'cuda:0': 'cpu'},
        }

    # model state
    model.load_state_dict(torch.load(checkpoint_path, **extra))

    # optimizer state
    if optimizer is not None:
        optimizer.load_state_dict(torch.load(
            checkpoint_path.replace('.ckpt', '.optim'), **extra))

    to_return = {
        'model': model,
        'optimizer': optimizer,
    }

    # extra metadata (eg epoch)
    metapath = checkpoint_path.replace('.ckpt', '.meta')
    if os.path.exists(metapath):
        meta = pickle.load(open(metapath, 'rb'))
        to_return.update(meta)

    return to_return


def _save_training_losses(
        trl, epoch: int, smoothing_window: int, grups_per_epoch: int,
        checkpoint_path: str):
    # save training loss -- by gradient update (smoothed)
    bygrup = DataFrame(
        columns=['epoch', 'gradient_update', 'training'] + [
            ltype for ltype in trl])
    for ltype in trl:
        bygrup.loc[:, ltype] = running_mean(
            list(trl[ltype].deque), smoothing_window)
    bygrup.loc[:, 'epoch'] = epoch
    # get gradient updates -- make sure it matches the smooth losses shape
    start = ((epoch - 1) * grups_per_epoch) + smoothing_window
    grups = [start]
    while len(grups) < bygrup.shape[0]:
        grups.append(grups[-1] + 1)
    bygrup.loc[:, 'gradient_update'] = grups
    bygrup.loc[:, 'training'] = True
    bygrup.to_csv(
        checkpoint_path.replace('.ckpt', '_trainingLossByGrUp.csv'),
        header=epoch == 1, mode='w' if epoch == 1 else 'a',
        index=False,
    )

    # save training loss -- by epoch
    byepoch = DataFrame(columns=list(bygrup.columns))
    for ltype in trl:
        byepoch.loc[:, ltype] = [trl[ltype].global_avg]
    byepoch.loc[:, 'epoch'] = epoch
    byepoch.loc[:, 'gradient_update'] = grups[-1]
    byepoch.loc[:, 'training'] = True
    byepoch.to_csv(
        checkpoint_path.replace('.ckpt', '_trainingLossByEpoch.csv'),
        header=epoch == 1, mode='w' if epoch == 1 else 'a',
        index=False,
    )


def _save_testing_metrics(
        tsl, epoch: int, checkpoint_path: str, grup: int, postfix=''):
    tsmetrics = DataFrame(
        columns=['epoch', 'gradient_update', 'training'] + [
            tmetric for tmetric in tsl])
    for tmetric, mval in tsl.items():
        tsmetrics.loc[:, tmetric] = [mval]
    tsmetrics.loc[:, 'epoch'] = epoch
    tsmetrics.loc[:, 'gradient_update'] = grup
    tsmetrics.loc[:, 'training'] = False
    tsmetrics.to_csv(
        checkpoint_path.replace(
            '.ckpt', f'_testingMetricsByEpoch{postfix}.csv'),
        header=epoch == 1, mode='w' if epoch == 1 else 'a',
        index=False,
    )


def _evaluate_on_testing_set(
        model, data_loader_test, device, test_maxDets, crop_inference_to_fov):
    evl_all, evl_objectness, classification_metrics = evaluate(
        model=model, data_loader=data_loader_test,
        device=device, maxDets=test_maxDets,
        crop_inference_to_fov=crop_inference_to_fov)

    tsl = {}
    classification = data_loader_test.dataset.do_classification  # noqa

    evls = [(evl_all, '')]
    if classification:
        evls.append((evl_objectness, 'objectness '))

    # Detection .. all or just objectness
    for evltuple in evls:

        evl, what = evltuple

        # by bounding box
        tsl.update({
            what + 'maxDets': test_maxDets[-1],
            what + 'mAP @ 0.50:0.95': evl.coco_eval['bbox'].stats[0],
            what + 'AP @ 0.5': evl.coco_eval['bbox'].stats[1],  # noqa
            what + 'AP @ 0.75': evl.coco_eval['bbox'].stats[2],  # noqa
        })
        # by mask
        if 'segm' in evl.coco_eval:
            tsl.update({
                what + 'segm mAP @ 0.50:0.95': evl.coco_eval['segm'].stats[0],
                # noqa
                what + 'segm AP @ 0.5': evl.coco_eval['segm'].stats[1],  # noqa
                what + 'segm AP @ 0.75': evl.coco_eval['segm'].stats[2],  # noqa
            })

    # just classification accuracy (for those correctly detected)
    if classification:
        tsl.update(classification_metrics)

    return tsl


def _get_optimizer(model, optimizer_type='SGD', optimizer_params=None):
    params = [p for p in model.parameters() if p.requires_grad]
    if optimizer_type == 'SGD':
        optimizer_params = {
            'lr': 0.005,
            'momentum': 0.9,
            'weight_decay': 0.0005,
        } if optimizer_params is None else optimizer_params
        optimizer = torch.optim.SGD(params, **optimizer_params)
    elif optimizer_type == 'Adam':
        optimizer_params = {
            'lr': 1e-4,
            'betas': (0.9, 0.999),
            'eps': 1e-8,
            'weight_decay': 0.0001,
        } if optimizer_params is None else optimizer_params
        optimizer = torch.optim.Adam(params, **optimizer_params)
    else:
        raise NotImplementedError(f'Unknown optimizer: {optimizer_type}')
    return optimizer


def _freeze_detection(
        model, optimizer_type='SGD', optimizer_params=None):
    """Freeze all non-classification layers"""

    def _alter(mod, requires_grad):
        for param in mod.parameters():
            param.requires_grad = requires_grad

    def _freeze(mod):
        _alter(mod, requires_grad=False)

    def _unfreeze(mod):
        _alter(mod, requires_grad=True)

    # freeze all model params
    _freeze(model)
    # unfreeze classification convolutions
    if model.roi_heads.cconvhead is not None:
        _unfreeze(model.roi_heads.cconvhead)
    # unfreeze self-attention head
    if model.roi_heads.sattention_head is not None:
        _unfreeze(model.roi_heads.sattention_head)
    # unfreeze FastRCNNPredictor EXCEPT bbox regression
    _unfreeze(model.roi_heads.box_predictor)
    _freeze(model.roi_heads.box_predictor.bbox_pred)

    # get optimizer, now only for params that require_grad
    # See: https://gist.github.com/L0SG/2f6d81e4ad119c4f798ab81fa8d62d3f#file-freeze_example-py-L74
    # IMPORTANT: pytorch optimizer explicitly accepts parameter that
    #  requires grad see https://github.com/pytorch/pytorch/issues/679
    optimizer = _get_optimizer(
        model=model, optimizer_type=optimizer_type,
        optimizer_params=optimizer_params)

    return model, optimizer


def trainNucleusModel(
        model, checkpoint_path: str,
        data_loader: DataLoader, data_loader_test: DataLoader = None,
        n_gradient_updates=100000, effective_batch_size=2,
        print_freq=1, window_size=None, smoothing_window=10,
        test_evaluate_freq=5, test_maxDets=None, crop_inference_to_fov=False,
        optimizer_type='SGD', optimizer_params=None,
        lr_scheduler_type=None, lr_scheduler_params=None,
        loss_weights=None, n_testtime_augmentations=None,
        freeze_det_after=None,
):
    """"""
    # some basic checks
    assert smoothing_window <= len(data_loader) // effective_batch_size
    test_maxDets = test_maxDets or [1, 100, 300]
    assert len(test_maxDets) == 3
    n_testtime_augmentations = n_testtime_augmentations or [0]
    freeze_det_after = freeze_det_after or np.inf

    # make sure max loss weight is 1
    if loss_weights is not None:
        mxl = max(loss_weights.values())
        loss_weights = {k: v / mxl for k, v in loss_weights.items()}

    # train on the GPU or on the CPU, if a GPU is not available
    device = torch.device('cuda') if torch.cuda.is_available() else \
        torch.device('cpu')

    # NOTE:
    #  The torch data parallelism loads all images into ONE GPU
    #  (the "main" one), copies the model on all GPUs, distributes the data to
    #  them, passes forward mode (which INCLUDES the loss in faster/maskrcnn),
    #  collects the output from all GPUs into the "main" GPU, then
    #  backprops the loss in parallel for all GPUs. See the diagram here:
    #  https://blog.paperspace.com/pytorch-memory-multi-gpu-debugging/
    #  What this means is that:
    #  1- All data needs to fit in one GPU EVEN THOUGH everything is parallel
    #  2- There's an imbalanced load on the GPUs, with the "main" GPU
    #  (gpu0, usually) handling more work & utilizing more memory than others.
    #  See this discussion:
    #  https://discuss.pytorch.org/t/dataparallel-imbalanced-memory-usage/22551
    #  Using DistributedDataParallel may help a bit in speed, but it does not
    #  help with the distibuted load thing .. everything STILL needs to fit to
    #  one GPU. I'll just train one fold per GPU.

    # GPU parallelize if gpus are available. See:
    #   https://pytorch.org/tutorials/beginner/blitz/ ...
    #   data_parallel_tutorial.html#create-model-and-dataparallel
    if NGPUS > 1:

        print(f"Let's use {NGPUS} GPUs!")
        model = DataParallel(model)

        # # DistributedDataParallel is next to impossible to get to work!
        # # I tried this, among MANY others:
        # # https://discuss.pytorch.org/t/use-distributed-data-parallel-correctly/82500/3
        # # I spent two full days trying to make it work, with all sorts of
        # # configs, with no luck. I give up!! It's not worth the time
        # # or effort; I'll just launch each fold independently
        # os.environ['MASTER_ADDR'] = 'localhost'
        # os.environ['MASTER_PORT'] = '29500'
        # torch.distributed.init_process_group(
        #     backend='nccl', init_method="env://", rank=0,
        #     world_size=torch.cuda.device_count(),  # use all visible gpus
        # )
        # model = DistributedDataParallel(model)

    # move model to the right device
    model.to(device)

    # # Mohamed: use half floats
    # if ISCUDA:
    #     model = model.to(torch.float16)

    # construct an optimizer
    optimizer = _get_optimizer(
        model=model, optimizer_type=optimizer_type,
        optimizer_params=optimizer_params)

    # load weights and optimizer state
    if os.path.exists(checkpoint_path):
        ckpt = load_ckp(
            checkpoint_path=checkpoint_path, model=model, optimizer=optimizer)
        model = ckpt['model']
        optimizer = ckpt['optimizer']
        start_epoch = ckpt['epoch']
        start_epoch += 1
    else:
        start_epoch = 1

    # learning rate scheduler
    if lr_scheduler_type is None:
        lr_scheduler = None
    elif lr_scheduler_type == 'step':
        lr_scheduler_params = lr_scheduler_params or {
            'step_size': 50,
            'gamma': 0.1,
        }
        grup = -1 if start_epoch == 1 else \
            (start_epoch - 1) * data_loader.batch_size
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer=optimizer, last_epoch=grup,
            **lr_scheduler_params
        )
    else:
        raise NotImplementedError(f'Unknown lr_scheduler: {lr_scheduler_type}')

    # keep track of all batch losses if window size is not given
    # otherwise, only keep track of the last window_size batches
    window_size = window_size or len(data_loader)

    # let's train it for n_epochs

    grups_per_epoch = int(len(data_loader.dataset) / effective_batch_size)
    n_epochs = int(n_gradient_updates // grups_per_epoch)

    frozen_det = False

    for epoch in range(start_epoch, n_epochs + 1):

        # Maybe freeze detection, but keep training classification
        if (not frozen_det) and (
                (epoch - 1) * grups_per_epoch > freeze_det_after):
            model, optimizer = _freeze_detection(
                model=model, optimizer_type=optimizer_type,
                optimizer_params=optimizer_params,
            )
            frozen_det = True

        # train for one epoch
        trl = train_one_epoch(
            model=model, device=device, epoch=epoch,
            optimizer=optimizer, lr_scheduler=lr_scheduler,
            data_loader=data_loader, effective_batch_size=effective_batch_size,
            loss_weights=loss_weights,
            print_freq=print_freq, window_size=window_size,
        )
        trl = {
            ltype: v for ltype, v in trl.meters.items()
            if ltype.startswith('loss')
        }

        # evaluate on the test dataset
        tsls = []
        if (data_loader_test is not None) and ((epoch == n_epochs) or (
                (epoch - 1) % test_evaluate_freq == 0)):

            # get performance at all requested testtime augmentation levels
            for ntta in n_testtime_augmentations:
                model.n_testtime_augmentations = ntta
                tsls.append(
                    _evaluate_on_testing_set(
                        model=model, data_loader_test=data_loader_test,
                        device=device, test_maxDets=test_maxDets,
                        crop_inference_to_fov=crop_inference_to_fov)
                )

        # save training loss
        _save_training_losses(
            trl=trl, epoch=epoch, smoothing_window=smoothing_window,
            grups_per_epoch=grups_per_epoch, checkpoint_path=checkpoint_path)

        # save testing metrics
        for i, tsl in enumerate(tsls):
            _save_testing_metrics(
                tsl=tsl, epoch=epoch, checkpoint_path=checkpoint_path,
                grup=epoch * grups_per_epoch,
                postfix = f'_{n_testtime_augmentations[i]}_TestTimeAugs',
            )

        # plot training and testing
        for i, ntta in enumerate(n_testtime_augmentations):
            pu.plot_accuracy_progress(
                checkpoint_path=checkpoint_path,
                postfix=f'_{ntta}_TestTimeAugs',
            )

        # create checkpoint and save
        print("*--- SAVING CHECKPOINT!! ---*")
        torch.save(model.state_dict(), f=checkpoint_path)
        torch.save(
            optimizer.state_dict(),
            f=checkpoint_path.replace('.ckpt', '.optim'))
        meta = {'epoch': epoch}
        with open(checkpoint_path.replace('.ckpt', '.meta'), 'wb') as f:
            pickle.dump(meta, f)


def evaluateNucleusModel(
        model, dloader: DataLoader, checkpoint_path=None, test_maxDets=None,
        n_testtime_augmentations=None, crop_inference_to_fov=False):

    test_maxDets = test_maxDets or [1, 100, 300]
    assert len(test_maxDets) == 3
    n_testtime_augmentations = n_testtime_augmentations or [0]

    # load weights and optimizer state
    if checkpoint_path is not None:
        ckpt = load_ckp(checkpoint_path=checkpoint_path, model=model)
        model = ckpt['model']

    # evaluate on the GPU or on the CPU, if a GPU is not available
    device = torch.device('cuda') if torch.cuda.is_available() else \
        torch.device('cpu')

    # get performance at all requested testtime augmentation levels
    tsls = []
    for ntta in n_testtime_augmentations:
        model.n_testtime_augmentations = ntta
        tsl  = _evaluate_on_testing_set(
            model=model, data_loader_test=dloader,
            device=device, test_maxDets=test_maxDets,
            crop_inference_to_fov=crop_inference_to_fov,
        )
        tsls.append(tsl)

    return tsls

