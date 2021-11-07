from os.path import join as opj
import numpy as np
from pandas import DataFrame, read_sql_query

from configs.nucleus_style_defaults import Interrater
from interrater.interrater_utils import calc_stats_simple, _get_clmap, \
    _connect_to_anchor_db, get_fovs_annotated_by_almost_everyone, \
    remap_classes_in_anchorsdf


def get_confusion_by_pathologist(
        dbcon, whoistruth, unbiased_is_truth, clsgroup, delete_existing):
    """"""
    assert Interrater.TRUTHMETHOD == 'EM', \
        "Only EM truth method supported because I aggregate probabs"

    clmap, class_list = _get_clmap(clsgroup)

    confusions = DataFrame(columns=[
        'unbiased_is_truth', 'whoistruth', 'evalset', 'participant',
        f'{Interrater.TRUTHMETHOD}_truth', 'undetected'] + class_list)

    truthcol = Interrater._get_truthcol(
        whoistruth=whoistruth, unbiased=unbiased_is_truth)
    ubstr = Interrater._ubstr(unbiased_is_truth)

    for evalset in Interrater.MAIN_EVALSET_NAMES:

        # read real anchors and remap labels
        tablename = f'v3.1_final_anchors_' \
                    f'{evalset}_{ubstr}{whoistruth}_AreTruth'
        all_real_anchors = read_sql_query(f"""
            SELECT * FROM "{tablename}"
        ;""", dbcon)

        # remap truth by aggregating EM probabs
        all_real_anchors = remap_classes_in_anchorsdf(
            anchors=all_real_anchors, clsgroup=clsgroup,
            also_ilabel=True, remove_ambiguous=True,
            who_determines_ambig=whoistruth, how_ambig_is_determined='EM',
        )

        # for each user, add confusions
        for usr in Interrater.All:

            print(
                f'{clsgroup.upper()}: '
                f'{ubstr}{whoistruth}: get_confusion_by_pathologist: '
                f'{evalset}: {usr}'
            )

            # read anchors
            keep = all_real_anchors.loc[:, usr] != 'DidNotAnnotateFOV'
            real_anchors = all_real_anchors.loc[keep, :].copy()
            fp_anchors = Interrater._query_fp_anchors_for_usr(
                dbcon=dbcon, whoistruth=whoistruth, unbiased=unbiased_is_truth,
                usr=usr, evalset=evalset)

            # remap false positive anchor labels
            fp_anchors = remap_classes_in_anchorsdf(
                anchors=fp_anchors, clsgroup=clsgroup, also_ilabel=False)

            # first we add a tally of the false positives
            row = {
                'unbiased_is_truth': unbiased_is_truth,
                'whoistruth': whoistruth,
                'evalset': evalset,
                'participant': usr,
                f'{Interrater.TRUTHMETHOD}_truth': 'non-existent'
            }
            row.update({
                cls: 0 for cls in ['undetected'] + class_list})
            uniq, cnts = np.unique(
                fp_anchors.loc[:, usr].values, return_counts=True)
            row.update({cls: int(cnt) for cls, cnt in zip(uniq, cnts)})
            confusions.loc[confusions.shape[0], :] = row

            # next we create a tally of confusions per each true label
            for truecls in class_list:
                row = {
                    'unbiased_is_truth': unbiased_is_truth,
                    'whoistruth': whoistruth,
                    'evalset': evalset,
                    'participant': usr,
                    f'{Interrater.TRUTHMETHOD}_truth': truecls}
                row.update(
                    {cls: 0 for cls in ['undetected'] + class_list})
                assigned_lbl = real_anchors.loc[real_anchors.loc[
                    :, truthcol] == truecls, usr]
                uniq, cnts = np.unique(assigned_lbl.values, return_counts=True)
                row.update({cls: int(cnt) for cls, cnt in zip(uniq, cnts)})
                confusions.loc[confusions.shape[0], :] = row

    # add table to db
    confusions.to_sql(
        name=f'confusion_byParticipant_{clsgroup}ClassGroup', con=dbcon,
        index=False, if_exists='replace' if delete_existing else 'append')


def get_confusion_by_anchor(
        dbcon, whoistruth, unbiased_is_truth, clsgroup, delete_existing):
    """"""
    clmap, class_list = _get_clmap(clsgroup)

    confusions = DataFrame(columns=[
       'unbiased_is_truth', 'whoistruth', 'evalset', 'who', 'n_fovs',
       'fovnames', 'N_who_AnnotatedFOV', 'n_anchors',
       f'{Interrater.TRUTHMETHOD}_truth', 'undetected',
    ] + class_list)

    truthcol = Interrater._get_truthcol(
        whoistruth=whoistruth, unbiased=unbiased_is_truth)

    for evalset in Interrater.MAIN_EVALSET_NAMES:
        for who in Interrater.CONSENSUS_WHOS:

            print(
                f'{clsgroup.upper()}: '
                f'{whoistruth}: get_confusion_by_anchor: {evalset}: {who}'
            )

            # restrict to relevant FOV subset and anchors
            out = get_fovs_annotated_by_almost_everyone(
                dbcon_anchors=dbcon, unbiased_is_truth=unbiased_is_truth,
                whoistruth=whoistruth, evalset=evalset, who=who)

            # group classes as needed
            anchs = remap_classes_in_anchorsdf(
                anchors=out['anchors'], clsgroup=clsgroup,
                also_ilabel=True, remove_ambiguous=True,
                who_determines_ambig=whoistruth, how_ambig_is_determined='EM',
            )

            # create a tally of confusions per each true label
            for truecls in class_list:

                assigned_lbl = anchs.loc[anchs.loc[
                    :, truthcol] == truecls,
                    Interrater.who[who]]
                row = {
                    'unbiased_is_truth': unbiased_is_truth,
                    'whoistruth': whoistruth,
                    'evalset': evalset,
                    'who': who,
                    'n_fovs': len(out['fovnames']),
                    'fovnames': ','.join(out['fovnames']),
                    'N_who_AnnotatedFOV': out['maxn'],
                    'n_anchors': assigned_lbl.shape[0],
                    f'{Interrater.TRUTHMETHOD}_truth': truecls
                }
                row.update(
                    {cls: 0. for cls in ['undetected'] + class_list})

                uniq, cnts = np.unique(assigned_lbl.values, return_counts=True)
                row.update({
                    cls: cnt / row['n_anchors'] for cls, cnt in zip(uniq, cnts)
                    if cls != 'DidNotAnnotateFOV'
                })

                confusions.loc[confusions.shape[0], :] = row

    # add table to db
    confusions.to_sql(
        name=f'confusion_byAnchor_{clsgroup}ClassGroup', con=dbcon,
        index=False, if_exists='replace' if delete_existing else 'append',
    )


def get_pathologist_accuracy_stats(
        dbcon, unbiased_is_truth, whoistruth, clsgroup, delete_existing):
    """"""
    _, class_list = _get_clmap(clsgroup)
    ubstr = Interrater._ubstr(unbiased_is_truth)
    # init
    stats = {
        evalset: {
            usr: {
                cls: {
                    'unbiased_is_truth': unbiased_is_truth,
                    'whoistruth': whoistruth,
                    'evalset': evalset,
                    'participant': usr,
                    'class': cls
                }
                for cls in ['detection', 'classification'] + class_list
            }
            for usr in Interrater.All
        } for evalset in Interrater.MAIN_EVALSET_NAMES
    }

    # get all stats
    for evalset in Interrater.MAIN_EVALSET_NAMES:
        for usr in Interrater.All:
            print(
                f'{clsgroup.upper()}: '
                f'{ubstr}{whoistruth}: get_pathologist_accuracy_stats: '
                f'{evalset}: {usr}'
            )

            # read confusion
            confusions = read_sql_query(f"""
                SELECT *
                FROM "confusion_byParticipant_{clsgroup}ClassGroup"
                WHERE "unbiased_is_truth" = {int(unbiased_is_truth)}
                  AND "whoistruth" = "{whoistruth}"
                  AND "evalset" = "{evalset}"
                  AND "participant" = "{usr}" 
            ;""", dbcon)

            # detection stats
            utruth = confusions.loc[:, f'{Interrater.TRUTHMETHOD}_truth']
            true_detection = confusions.loc[
                utruth != 'non-existent', class_list]
            true_detection.index = [j for j in utruth if j != 'non-existent']
            stats[evalset][usr]['detection'].update(calc_stats_simple(
                TP=np.sum(true_detection.values),
                FP=np.sum(confusions.loc[
                    utruth == 'non-existent', class_list].values),
                FN=np.sum(confusions.loc[
                    utruth != 'non-existent', 'undetected']),
            ))

            # classification stats by class
            aggregate = {k: 0 for k in ['TP', 'FP', 'TN', 'FN']}
            sclasses = set(class_list)
            for cls in class_list:
                st = {
                    'TP': true_detection.loc[cls, cls],
                    'FP': np.sum(true_detection.loc[
                        sclasses.difference({cls}), cls].values),
                    'TN': np.sum(true_detection.loc[
                        sclasses.difference({cls}),
                        sclasses.difference({cls})].values),
                    'FN': np.sum(true_detection.loc[
                        cls, sclasses.difference({cls})].values),
                }
                stats[evalset][usr][cls].update(calc_stats_simple(**st))
                aggregate.update({
                    k: aggregate[k] + v for k, v in st.items()
                })

            # aggregate classification stats
            stats[evalset][usr]['classification'].update(
                calc_stats_simple(**aggregate))

    # convert to df and commit to db
    records = DataFrame.from_records([
        stats[evalset][usr][cls]
        for evalset in Interrater.MAIN_EVALSET_NAMES
        for usr in Interrater.All
        for cls in ['detection', 'classification'] + class_list
    ])
    keep = records.loc[:, 'total'] > 1
    records = records.loc[keep, :]
    records.to_sql(
        name=f'participant_AccuracyStats_{clsgroup}ClassGroup', con=dbcon,
        index=False, if_exists='replace' if delete_existing else 'append',
    )

# %%===========================================================================


def main():

    DATASETNAME = 'CURATED_v1_2020-03-29_EVAL'

    # where to save stuff
    BASEPATH = "/home/mtageld/Desktop/cTME/results/tcga-nucleus/interrater/"
    SAVEPATH = opj(BASEPATH, DATASETNAME, 'i1_anchors')

    # connect to database
    dbcon = _connect_to_anchor_db(opj(SAVEPATH, '..'))

    for clsgroup in ['main', 'super']:

        delete_existing = True

        for unbiased_is_truth in [False]:
            for whoistruth in ['Ps']:  # Interrater.CONSENSUS_WHOS:

                    common_params = {
                        'dbcon': dbcon,
                        'whoistruth': whoistruth,
                        'unbiased_is_truth': unbiased_is_truth,

                        'clsgroup': clsgroup,
                        'delete_existing': delete_existing,
                    }
                    delete_existing = False

                    # get confusion matrices by pathologist
                    get_confusion_by_pathologist(**common_params)

                    # get confusion matrices by anchor
                    get_confusion_by_anchor(**common_params)

                    # get accuracy stats
                    get_pathologist_accuracy_stats(**common_params)

# %%===========================================================================


if __name__ == '__main__':
    main()
