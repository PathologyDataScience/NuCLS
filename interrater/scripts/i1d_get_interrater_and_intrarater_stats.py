from os.path import join as opj
import numpy as np
from pandas import DataFrame, concat, read_sql_query
from itertools import combinations
from sklearn.metrics import cohen_kappa_score

from configs.nucleus_style_defaults import Interrater
from interrater.interrater_utils import _connect_to_anchor_db, \
    remap_classes_in_anchorsdf


def _read_and_remap_anchors(
        dbcon, evalset, clsgroup,
        unbiased_is_truth=False, whoistruth='Ps'):

    ubstr = Interrater._ubstr(unbiased_is_truth)

    # read real anchors and remap labels
    tablename = f'v3.1_final_anchors_' \
                f'{evalset}_{ubstr}{whoistruth}_AreTruth'
    all_real_anchors = read_sql_query(f"""
        SELECT * FROM "{tablename}"
    ;""", dbcon)

    # remap truth  .. the only reason we need truth is to exclude "false"
    # anchors that were inferred (from pathologist annotations) to be
    # ambiguous and therefore are not reliable for measuring concordance etc
    all_real_anchors = remap_classes_in_anchorsdf(
        anchors=all_real_anchors, clsgroup=clsgroup,
        also_ilabel=True, remove_ambiguous=True,
        who_determines_ambig=whoistruth, how_ambig_is_determined='EM',
    )
    all_real_anchors.index = all_real_anchors.loc[:, 'anchor_id']

    return all_real_anchors


def get_intrarater_stats(dbcon, clsgroup):
    """"""
    ir = Interrater
    stats = DataFrame(columns=[
       'participant', 'first_evalset', 'second_evalset',
       'n_anchors_all', 'n_anchors_intersection',
       'detection_and_classification', 'classification',
       'n_anchors_first_evalset', 'n_anchors_second_evalset',
       'n_clicks_first_evalset', 'n_clicks_second_evalset',
    ])

    def _get_anchors(real_anchors, pathol, evalset):
        anch = real_anchors.loc[
            real_anchors.loc[:, pathol] != 'DidNotAnnotateFOV',
            [pathol, 'algorithmic_clicks_All']].copy()
        anch.columns = [evalset, f'{evalset}_click']
        anch.loc[:, f'{evalset}_click'] = anch.loc[
            :, f'{evalset}_click'].apply(lambda x: 0 + (usr in x))
        return anch

    # The U-control set is what other sets are compared to
    ev1 = 'U-control'
    ev2s = set(ir.MAIN_EVALSET_NAMES).difference({'U-control'})

    # read all "real" anchors for the first evalset
    real_anchors1 = _read_and_remap_anchors(
        dbcon=dbcon, evalset=ev1, clsgroup=clsgroup)

    for ev2 in ev2s:

        # read all "real" anchors for the second evalset
        real_anchors2 = _read_and_remap_anchors(
            dbcon=dbcon, evalset=ev2, clsgroup=clsgroup)

        for usr in ir.All:

            print(f'{clsgroup.upper()}: get_intrarater_stats: {usr}')

            # concat relevant anchors for user
            anchors1 = _get_anchors(
                real_anchors=real_anchors1, pathol=usr, evalset=ev1)
            anchors2 = _get_anchors(
                real_anchors=real_anchors2, pathol=usr, evalset=ev2)
            anchors = concat(
                (anchors1, anchors2), axis=1,
                join='inner', ignore_index=False)

            if anchors.shape[0] == 0:
                continue

            # detection and classification
            row = {
                'participant': usr,
                'first_evalset': ev1,
                'second_evalset': ev2,
                'n_anchors_all': int(anchors.shape[0]),
                'detection_and_classification': cohen_kappa_score(
                    anchors.loc[:, ev1], anchors.loc[:, ev2]),
                'n_anchors_first_evalset': int(anchors.loc[
                    anchors.loc[:, ev1] != 'undetected', :].shape[0]),
                'n_anchors_second_evalset': int(anchors.loc[
                    anchors.loc[:, ev2] != 'undetected', :].shape[0]),
                'n_clicks_first_evalset':  int(np.nansum(
                    anchors.loc[:, f'{ev1}_click'].values)),
                'n_clicks_second_evalset': int(np.nansum(
                    anchors.loc[:, f'{ev2}_click'].values)),
            }

            # classification only -- keep if detected in both evalsets
            keep = np.sum(0 + (
                anchors.loc[:, [ev1, ev2]] == 'undetected'), axis=1) == 0
            anchors = anchors.loc[keep, :]
            row.update({
                'classification': cohen_kappa_score(
                    anchors.loc[:, ev1], anchors.loc[:, ev2]),
                'n_anchors_intersection': int(anchors.shape[0]),
            })
            stats.loc[stats.shape[0], :] = row

    # add table to db
    stats.to_sql(
        name=f'intra-rater_{clsgroup}ClassGroup',
        con=dbcon, index=False, if_exists='replace')

# %%===========================================================================


def get_interrater_stats(dbcon, clsgroup):
    """"""
    ir = Interrater
    stats = DataFrame(columns=[
        'evalset', 'first_participant', 'second_participant',
        'n_anchors_all', 'n_anchors_intersection',
        'detection_and_classification', 'classification',
    ])

    # go through various eval sets
    for evalset in ir.MAIN_EVALSET_NAMES:

        # read all "real" anchors for this evalset
        eval_anchors = _read_and_remap_anchors(
            dbcon=dbcon, evalset=evalset, clsgroup=clsgroup)
        eval_anchors = eval_anchors.loc[:, ir.who['All']]

        for usr1, usr2 in combinations(list(eval_anchors.columns), 2):

            print(
                f'{clsgroup.upper()}: get_inter-rater_stats: {evalset}: '
                f'{usr1} VS. {usr2}')

            # restrict to FOVs annotated by both
            anchors = eval_anchors.loc[:, [usr1, usr2]]
            keep1 = np.sum(0 + anchors.isnull(), axis=1) == 0
            keep2 = np.sum(0 + (anchors == 'DidNotAnnotateFOV'), axis=1) == 0
            anchors = anchors.loc[keep1, :]
            anchors = anchors.loc[keep2, :]

            # # detection & classification -- keep if detected in 1+ evalset
            # keep = np.sum(0 + (anchors == 'undetected'), axis=1) < 2
            # anchors = anchors.loc[keep, :]

            if anchors.shape[0] == 0:
                continue

            # detection and classification
            row = {
                'evalset': evalset,
                'first_participant': usr1,
                'second_participant': usr2,
                'n_anchors_all': int(anchors.shape[0]),
                'detection_and_classification': cohen_kappa_score(
                    anchors.loc[:, usr1], anchors.loc[:, usr2]),
            }

            # classification only -- keep if detected in both evalsets
            keep = np.sum(0 + (anchors == 'undetected'), axis=1) == 0
            anchors = anchors.loc[keep, :]
            row.update({
                'classification': cohen_kappa_score(
                    anchors.loc[:, usr1], anchors.loc[:, usr2]),
                'n_anchors_intersection': int(anchors.shape[0]),
            })
            stats.loc[stats.shape[0], :] = row

    # add table to db
    stats.to_sql(
        name=f'inter-rater_{clsgroup}ClassGroup',
        con=dbcon, index=False, if_exists='replace')


# %%===========================================================================


def main():

    DATASETNAME = 'CURATED_v1_2020-03-29_EVAL'

    # where to save stuff
    BASEPATH = "/home/mtageld/Desktop/cTME/results/tcga-nucleus/interrater/"
    SAVEPATH = opj(BASEPATH, DATASETNAME, 'i1_anchors')

    # connect to database
    dbcon = _connect_to_anchor_db(opj(SAVEPATH, '..'))

    # get stats
    for clsgroup in ['main', 'super']:
        get_intrarater_stats(dbcon, clsgroup=clsgroup)
        get_interrater_stats(dbcon, clsgroup=clsgroup)


# %%===========================================================================

if __name__ == '__main__':
    main()
