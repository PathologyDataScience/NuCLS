import os
from os.path import join as opj

import numpy as np
from histomicstk.annotations_and_masks.annotation_and_mask_utils import \
    np_vec_no_jit_iou
from scipy.optimize import linear_sum_assignment
from pandas import DataFrame, read_csv, concat

from GeneralUtils import load_configs  # noqa


def load_saved_otherwise_default_model_configs(configs_path,
                                               warn=True, purec=False):
    if os.path.exists(configs_path):
        if warn:
            input(f"Loading existing configs from: {configs_path}. Continue?")
        cfg = load_configs(configs_path=configs_path)
    else:
        print("Loading default configs")
        if not purec:
            import ctme.configs.nucleus_model_configs as cfg
        else:
            import ctme.configs.pure_classification_model_configs as cfg
    return cfg


def map_bboxes_using_hungarian_algorithm(bboxes1, bboxes2, min_iou=1e-4):
    """Map bounding boxes using hungarian algorithm.

    Adapted from Lee A.D. Cooper.

    Parameters
    ----------
    bboxes1 : numpy array
        columns correspond to xmin, ymin, xmax, ymax

    bboxes2 : numpy array
        columns correspond to xmin, ymin, xmax, ymax

    min_iou : float
        minumum iou to match two bboxes to match to each other

    Returns
    -------
    np.array
        matched indices relative to x1, y1

    np.array
        matched indices relative to x2, y2, correspond to the first output

    np.array
        unmatched indices relative to x1, y1

    np.array
        unmatched indices relative to x2, y2

    """
    # generate cost matrix for mapping cells from user to anchors
    max_cost = 1 - min_iou
    costs = 1 - np_vec_no_jit_iou(bboxes1=bboxes1, bboxes2=bboxes2)
    costs[costs > max_cost] = 99.

    # perform hungarian algorithm mapping
    source, target = linear_sum_assignment(costs)

    # discard mappings that are non-allowable
    allowable = costs[source, target] <= max_cost
    source = source[allowable]
    target = target[allowable]

    # find indices of unmatched
    def _find_unmatched(coords, matched):
        potential = np.arange(coords.shape[0])
        return potential[~np.in1d(potential, matched)]
    unmatched1 = _find_unmatched(bboxes1, source)
    unmatched2 = _find_unmatched(bboxes2, target)

    return source, target, unmatched1, unmatched2


def _get_tsms_for_coreset(model_root, model_name):
    tsm_str = 'testingMetricsByEpoch_0_TestTimeAugs'
    folds = list(range(1, 6))
    tsms = [
        read_csv(
            opj(model_root, f'fold_{fold}', f'{model_name}_{tsm_str}.csv'),
            index_col=0
        ).iloc[[-1]] for fold in folds
    ]
    tsms = concat(tsms, axis=0, ignore_index=True)
    tsms.loc[:, 'fold'] = folds
    tsms.loc[:, 'Testing Set'] = np.nan
    tsms.loc[:, 'Super-classes'] = np.nan
    return tsms


def _get_tsms_for_evalset(model_root, whoistruth='Ps', evalset='E'):
    tsms = []
    tsmp = opj(f'Eval_{whoistruth}AreTruth_{evalset}', 'testingMetrics_0.txt')
    folds = list(range(1, 6))
    for fold in folds:
        if fold == 2:
            tsm = tsms[-1].copy()
            tsm[:] = np.nan
            tsms.append(tsm)
            continue
        with open(opj(model_root, f'fold_{fold}', f'{tsmp}')) as f:
            tsm = {
                l.split("'")[1]: np.float32(l.split("'")[2].split(': ')[-1])
                for l in f.read().splitlines()
            }
            tsms.append(DataFrame(tsm, index=[0]))
    tsms = concat(tsms, axis=0, ignore_index=True)
    tsms.loc[:, 'fold'] = folds
    tsms.loc[:, 'Testing Set'] = np.nan
    tsms.loc[:, 'Super-classes'] = np.nan
    if evalset == 'U-control':
        tsms.loc[:, 'seg_n'] = np.nan
        tsms.loc[:, 'seg_medIOU'] = np.nan
        tsms.loc[:, 'seg_medDICE'] = np.nan
    return tsms
