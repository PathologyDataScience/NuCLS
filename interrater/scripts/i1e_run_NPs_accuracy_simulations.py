from os.path import join as opj
import numpy as np
from pandas import DataFrame, read_sql_query, concat, Series

from configs.nucleus_style_defaults import Interrater as ir
from interrater.interrater_utils import calc_stats_simple, \
    _connect_to_anchor_db, get_roc_and_auroc_for_who, \
    remap_classes_in_anchorsdf, _get_clmap
from interrater.DawidAndSkene1979_EMGtruthInference import EM, gete2wlandw2el_fromdf

# for reproducibility
np.random.seed(0)


def _get_fovdf(
        dbcon, evalset: str, min_ps_per_fov: int, min_fovs_per_p: int):
    # get fov metadata
    fovmetas = read_sql_query(f"""
            SELECT  "fovname", "participants_{evalset}" AS "participants"
            FROM "fov_meta"
            WHERE "participants_{evalset}" NOT NULL
        ;""", dbcon)
    fovnames = list(set(fovmetas.loc[:, 'fovname']))
    fovmetas.index = fovmetas.loc[:, 'fovname']

    # init dataframe of fovs and who annotated them
    fovdf = DataFrame(0, index=fovnames, columns=ir.NPs)
    for fovname, row in fovmetas.iterrows():
        for p in row['participants'].split(','):
            if p in ir.NPs:
                fovdf.loc[fovname, p] = 1

    # only keep participants and fovs if > a certain threshold
    fovdf = fovdf.loc[:, fovdf.sum(axis=0) >= min_fovs_per_p]
    fovdf = fovdf.loc[fovdf.sum(axis=1) >= min_ps_per_fov, :]

    return fovdf, fovnames


def _query_anchors_for_sim(
        dbcon, evalset: str, fovnames: list, usrs: list, clsgroup: str):
    """"""
    # query "real" anchors
    common = f"""
            "min_iou" = {ir.CMINIOU}
        AND "fovname" IN ({ir._get_sqlitestr_for_list(fovnames)})
    """
    all_anchors = read_sql_query(f"""
        SELECT *
        FROM "all_anchors_{evalset}"
        WHERE {common}
    ;""", dbcon)
    all_anchors.index = all_anchors.loc[:, 'anchor_id']
    all_anchors.drop('anchor_id', axis=1, inplace=True)

    print("    Remapping anchor labels ...")
    all_anchors = remap_classes_in_anchorsdf(
        all_anchors, clsgroup=clsgroup, also_ilabel=True,
        remove_ambiguous=True, who_determines_ambig='Ps',
        how_ambig_is_determined='EM')

    # refactor columns
    truthcol = ir._get_truthcol(whoistruth='Ps', unbiased=False)
    all_anchors = all_anchors.loc[:, ['fovname', truthcol] + usrs]
    all_anchors.rename(columns={truthcol: 'EMtruth_Ps'}, inplace=True)

    # flag "false" anchors, which were either detected
    # by just one P or the consensus was "undetected"
    all_anchors.loc[:, 'is_real'] = True
    all_anchors.loc[
        all_anchors.loc[:, "EMtruth_Ps"].isnull(), 'is_real'] = False
    all_anchors.loc[
        all_anchors.loc[:, "EMtruth_Ps"] == 'undetected', 'is_real'] = False

    return all_anchors


def _get_simfov_realization(fovdf: DataFrame, evalset: str, pperfov: int):
    """"""
    simfovs = []
    unique_ps = set()

    for fovname, row in fovdf.iterrows():
        potential_ps = list(row.index[row == 1])

        if len(potential_ps) < pperfov:
            raise Exception('More simulated NPs asked than are available!')

        selected_ps = np.random.choice(
            potential_ps, size=pperfov, replace=False).tolist()
        simfovs.append({
            'evalset': evalset,
            'fovname': fovname,
            'ps_per_fov': pperfov,
        })
        simfovs[-1].update({
            p: 1 if p in selected_ps else 0 for p in fovdf.columns
        })
        unique_ps = unique_ps.union(set(selected_ps))

    simfovs = DataFrame.from_records(simfovs)
    n_unique_ps = len(unique_ps)

    return simfovs, n_unique_ps


def _decimate_didntannotatefov(
        all_anchors: DataFrame, simfov: DataFrame, usrs: list):
    """"""
    anchors = []
    fovnames = np.unique(simfov.loc[:, 'fovname']).tolist()
    for fovname in fovnames:
        # decimate participants who didn't annotate this FOV
        # (or rather, simulated to not have annotated it)
        anch = all_anchors.loc[all_anchors.loc[
            :, 'fovname'] == fovname, :].copy()
        didntannotate = [
            k for k, v in dict(simfov.loc[simfov.loc[
                :, 'fovname'] == fovname, usrs].iloc[0, :]).items()
            if v == 0
        ]
        anch.loc[:, didntannotate] = 'DidNotAnnotateFOV'
        anchors.append(anch)
    anchors = concat(anchors, axis=0)
    return anchors


def _add_inferred_EM_label_for_usrs(
        anchors: DataFrame, usrs: list, class_list: list,
        initquality: float = 0.7, iterations: int = 70,
        add_class_probs: bool = True) -> DataFrame:
    """"""
    # convert to needed format
    tmpanchors = anchors.loc[:, usrs].copy()

    if ir.UNDETECTED_IS_A_CLASS:
        # option 1: undetected is a conscious decision
        e2wl, w2el, label_set = gete2wlandw2el_fromdf(
            df=tmpanchors, missingval='DidNotAnnotateFOV')
    else:
        # option 2: Undetected is a "missing" value
        tmpanchors[tmpanchors == 'DidNotAnnotateFOV'] = 'undetected'  # noqa
        e2wl, w2el, label_set = gete2wlandw2el_fromdf(
            df=tmpanchors, missingval='undetected')

    # Run D&S EM algorithm
    em = EM(
        e2wl=e2wl, w2el=w2el, label_set=label_set,
        initquality=initquality)
    e2lpd, w2cm = em.Run(iterr=iterations)

    ilprobs = DataFrame.from_dict(e2lpd, orient='index')
    ilabel = Series(np.argmax(ilprobs.values, axis=1))
    ilabel = ilabel.map(
        {k: v for k, v in enumerate(list(ilprobs.columns))})
    ilabelconf = ilprobs.max(axis=1)

    anchors.loc[:, f'ilabel'] = ilabel.tolist()
    anchors.loc[:, f'ilabel_confidence'] = ilabelconf

    if add_class_probs:
        for cls in ['undetected'] + class_list:
            if cls in ilprobs.columns:
                anchors.loc[:, f'iprob_{cls}'] = \
                    ilprobs.loc[:, cls]
            else:
                anchors.loc[:, f'iprob_{cls}'] = 0.

    return anchors


def _get_confusion_for_inferred_label(
        anchors: DataFrame, truthcol: str, class_list: list):
    """"""
    # separate true and false anchors
    tmp = anchors.copy()
    tmp.index = tmp.loc[:, f'{truthcol}']
    tmp = tmp.loc[:, ['ilabel', 'is_real']]
    real_anchors = tmp.loc[tmp.loc[:, 'is_real'], 'ilabel']
    fp_anchors = tmp.loc[~tmp.loc[:, 'is_real'], 'ilabel']
    fp_anchors = fp_anchors[fp_anchors != 'undetected']

    confusions = DataFrame(columns=['undetected'] + class_list)

    # first we add a tally of the false positives
    row = {cls: 0 for cls in ['undetected'] + class_list}
    uniq, cnts = np.unique(fp_anchors.values, return_counts=True)
    row.update({cls: int(cnt) for cls, cnt in zip(uniq, cnts)})
    confusions.loc['non-existent', :] = row

    # next we create a tally of confusions per each true label
    for truecls in class_list:
        assigned_lbl = real_anchors[real_anchors.index == truecls]
        row = {cls: 0. for cls in ['undetected'] + class_list}
        uniq, cnts = np.unique(assigned_lbl.values, return_counts=True)
        row.update({cls: int(cnt) for cls, cnt in zip(uniq, cnts)})
        confusions.loc[truecls, :] = row

    return confusions


def _get_stats_for_inferred_label(
        anchors: DataFrame, truthcol: str, class_list: list):
    """"""
    stats = {}

    # get area under classification ROC curve
    _, _, auroc = get_roc_and_auroc_for_who(
        anchors=anchors.loc[anchors.loc[:, 'is_real'], :],
        truthcol=truthcol, probcol_prefix='iprob_', probcol_postfix='')

    stats.update({f'auroc-{cls}': v for cls, v in auroc.items()})

    # get confusion matrix (inferred label compared to Ps "truth")
    confusions = _get_confusion_for_inferred_label(
        anchors=anchors, truthcol=truthcol, class_list=class_list)

    # detection stats
    utruth = confusions.index
    true_detection = confusions.loc[
        utruth != 'non-existent', class_list]
    stats.update({
        f'detection-{m}': v
        for m, v in calc_stats_simple(
            TP=np.sum(true_detection.values),
            FP=np.sum(confusions.loc[
                utruth == 'non-existent', class_list].values),
            FN=np.sum(confusions.loc[
                utruth != 'non-existent', 'undetected']),
        ).items()
    })

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
        stats.update({
            f'classification-{cls}-{m}': v
            for m, v in calc_stats_simple(**st).items()
        })
        aggregate.update({
            k: aggregate[k] + v for k, v in st.items()
        })

    # aggregate classification stats
    stats.update({
        f'classification-all-{m}': v
        for m, v in calc_stats_simple(**aggregate).items()
    })

    return stats


def _get_fovdf_and_anchors(
        dbcon, evalset: str, clsgroup: str,
        min_fovs_per_p: int, min_ps_per_fov: int):
    """"""
    # dataframe of fovs and who annotated them
    fovdf, fovnames = _get_fovdf(
        dbcon=dbcon, evalset=evalset, min_fovs_per_p=min_fovs_per_p,
        min_ps_per_fov=min_ps_per_fov)

    # query anchors
    usrs = list(fovdf.columns)
    all_anchors = _query_anchors_for_sim(
        dbcon, evalset=evalset, fovnames=fovnames, usrs=usrs,
        clsgroup=clsgroup)

    return fovdf, all_anchors


def _sim(
        simid, evalset: str, pperfov: int, class_list: list,
        ev_fovdf: DataFrame, ev_anchors: DataFrame):
    """"""
    # "Assign" FOVs to participants up to desired max_ps_per_fov
    simfov, n_unique_nps = _get_simfov_realization(
        fovdf=ev_fovdf, evalset=evalset, pperfov=pperfov)

    # ignore participants who were simulated to NOT have annotated
    # this fov, for each fov
    usrs = list(ev_fovdf.columns)
    anchors = _decimate_didntannotatefov(
        all_anchors=ev_anchors, simfov=simfov, usrs=usrs)

    # now infer label
    anchors = _add_inferred_EM_label_for_usrs(
        anchors=anchors, usrs=usrs, class_list=class_list)

    # get stats and append result
    row = {
        'simid': simid,
        'whoistruth': 'Ps',
        'evalset': evalset,
        'NPs_per_fov': pperfov,
        'n_unique_NPs': n_unique_nps,
    }
    row.update(_get_stats_for_inferred_label(
        anchors=anchors, truthcol='EMtruth_Ps', class_list=class_list)
    )
    return row


def get_single_simulation_result(
        dbcon, evalset: str, simidx: int, replace: bool,
        min_ps_per_fov: int, max_sim_ps_per_fov: int, clsgroup: str,
        min_sim_ps_per_fov: int = 2, min_fovs_per_p: int = 1):
    """"""
    _, class_list = _get_clmap(clsgroup)

    simid = simidx
    # simid = '%032x' % random.getrandbits(128)

    simresults = []

    # get anchors and fov info for evalset
    fovdf, ev_anchors = _get_fovdf_and_anchors(
        dbcon=dbcon, evalset=evalset, clsgroup=clsgroup,
        min_fovs_per_p=min_fovs_per_p, min_ps_per_fov=min_ps_per_fov)

    for pperfov in range(min_sim_ps_per_fov, max_sim_ps_per_fov + 1):

        print(f'sim {simidx}: {evalset}: NPs_per_fov={pperfov}')

        simresults.append(
            _sim(
                simid=simid, evalset=evalset, class_list=class_list,
                pperfov=pperfov, ev_fovdf=fovdf, ev_anchors=ev_anchors,
            )
        )

    # save to disk
    print(f'sim {simidx}: {evalset}: SAVING!')
    simresults = DataFrame.from_records(simresults)
    simresults.to_sql(
        name=f'NPs_AccuracySimulations_{clsgroup}ClassGroup',
        con=dbcon, index=False,
        if_exists='replace' if replace else 'append',
    )


def simulations(
        nsims: int, min_ps_per_fov: int, max_sim_ps_per_fov: int,
        **kwargs):
    """Run experiments to answer the following question: How many NPs
    do we need to obtain-per-FOV (keeping the total number of NPs who
    participate in the study constant) in order for their combined inferred
    label to match the truth we obtained from all pathologists. i.e. This is
    a question about how much redundancy do we need per fov.
    """
    assert max_sim_ps_per_fov < min_ps_per_fov, \
        'No of minimum available NPs per FOV must be higher than the' \
        'maximum simulated NPs per FOV to ensure fair comparison'

    replace = True
    for simidx in range(nsims):
        for evalset in ['E']:
            for clsgroup in ['super']:  # ['main', 'super']
                get_single_simulation_result(
                    evalset=evalset, simidx=simidx, replace=replace,
                    clsgroup=clsgroup,
                    min_ps_per_fov=min_ps_per_fov,
                    max_sim_ps_per_fov=max_sim_ps_per_fov,
                    **kwargs
                )
                replace = False

# %%===========================================================================


def main():

    DATASETNAME = 'CURATED_v1_2020-03-29_EVAL'

    # where to save stuff
    BASEPATH = "/home/mtageld/Desktop/cTME/results/tcga-nucleus/interrater/"
    SAVEPATH = opj(BASEPATH, DATASETNAME, 'i1_anchors')

    # connect to database
    dbcon = _connect_to_anchor_db(opj(SAVEPATH, '..'))

    # run the experiement for various evalsets
    simulations(
        dbcon=dbcon, nsims=1000,
        min_ps_per_fov=16, max_sim_ps_per_fov=15,
    )


# %%===========================================================================


if __name__ == '__main__':
    main()
