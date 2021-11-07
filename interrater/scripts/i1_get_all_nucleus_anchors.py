from os.path import join as opj
import numpy as np
import json
from pandas import DataFrame, Series, read_sql_query
from sqlalchemy import create_engine
from configs.nucleus_style_defaults import Interrater as ir
from interrater.interrater_utils import (
    _connect_to_anchor_db, get_fovinfos_for_interrater,
    get_anchors_for_iou_thresh, _maybe_mkdir
)
from interrater.DawidAndSkene1979_EMGtruthInference import EM, gete2wlandw2el_fromdf

# NOTE: CandygramAPI is a private API that connects to a girder client
# The class contains private access tokens that cannot be shared publicly.
from configs.ACS_configs import CandygramAPI  # noqa


def get_all_nucleus_anchors_gtruth(
        fovinfos, get_medoids_kwargs, dbcon, fovs_to_use=None,
        min_ious=None, constrained=True):
    """Get medoids for all fovs and save to a sqlite db.

    Parameters
    ----------
    fovinfos: dict
    get_medoids_kwargs: dict
    dbcon: object
    fovs_to_use: list or None
    min_ious: None
    constrained: bool

    Returns
    -------
    None

    """
    fovnames = list(fovinfos.keys())
    if fovs_to_use is None:
        fovs_to_use = fovnames
    if min_ious is None:
        min_ious = np.arange(0.05, 0.76, 0.05)

    for fovno, fovname in enumerate(fovs_to_use):
        fovinfo = fovinfos[fovname]
        for min_iou in min_ious:

            print(
                f'{"" if constrained else "unconstrained"}: '
                f'fov {fovno} of {len(fovs_to_use)}: {fovname}'
                ': min_iou= %.2f' % min_iou
            )

            # get and save the cluster medoids
            get_medoids_kwargs.update({
                'min_iou': min_iou,
                'fovinfo': fovinfo,
                'constrained': constrained,
            })
            out = get_anchors_for_iou_thresh(**get_medoids_kwargs)

            # handle empty eval sets -- return None
            if out is None:
                continue

            # add fov metadata to sqlite
            if min_iou == min_ious[0]:
                fovmeta = {
                    'fovname': fovname,
                    'fov_type': fovinfos[fovname]['fov_type'],
                    'slide_name': fovname.split('_')[0],
                    'slide_id': out['slide_id'],
                }
                anyevset = list(fovinfo.keys())[0]
                anyp = list(fovinfo[anyevset].keys())[0]
                fovcontloc = {
                    loc: fovinfo[anyevset][anyp][loc]
                    for loc in ['maybe-xmin', 'maybe-ymin',
                                'maybe-xmax', 'maybe-ymax']
                }
                for evset in ir.EVALSET_NAMES:
                    if evset in fovinfo.keys():
                        fovmeta[f'fov_ids_{evset}'] = ",".join([
                            str(p['fov_id'])
                            for _, p in fovinfo[evset].items()
                        ])
                        fovmeta[f'participants_{evset}'] = ",".join(
                            list(fovinfo[evset].keys()))
                    else:
                        fovmeta[f'fov_ids_{evset}'] = np.nan
                        fovmeta[f'participants_{evset}'] = np.nan
                    fovmeta.update(fovcontloc)

                fovmeta.update(out['bounds'])
                fovmeta = DataFrame(fovmeta, index=[0])
                if constrained:
                    fovmeta.to_sql(
                        name=f'fov_meta', con=dbcon,
                        if_exists='append', index=False)

            # add anchors to sqlite
            for evset, anchors in out['all_anchors'].items():
                if anchors.shape[0] < 1:
                    continue
                anchors.loc[:, 'fovname'] = fovname

                # make sure anchor ids are globally unique across fovs
                anchors.loc[:, 'anchor_id'] = anchors.loc[
                    :, 'anchor_id'].apply(lambda x: f'{x}_{fovname}')

                # add unbiased n_matches
                for who in ir.CONSENSUS_WHOS:
                    coln = f'n_matches_{who}'
                    anchors.loc[:, f'UNBIASED_{coln}'] = out['all_anchors'][
                        'U-control'][f'{coln}']

                anchors.to_sql(
                    name=f'all_anchors_{evset}', con=dbcon,
                    if_exists='append', index=False)

# %%===========================================================================


def _add_EM_for_detection_and_classification(
        anchors: DataFrame, who: str, initquality: float = 0.7,
        iterations: int = 70) -> DataFrame:
    """"""
    # convert to needed format
    usrs = ir.who[who]
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

    anchors.loc[:, f'EM_inferred_label_{who}'] = ilabel
    anchors.loc[:, f'EM_inferred_label_confidence_{who}'] = \
        ilabelconf
    for cls in ['undetected'] + ir.CLASSES:
        if cls in ilprobs.columns:
            anchors.loc[:, f'EM_prob_{cls}_{who}'] = \
                ilprobs.loc[:, cls]
        else:
            anchors.loc[:, f'EM_prob_{cls}_{who}'] = 0.

    # find out how many agree with the inferred ("true") label
    for lbl in np.unique(ilabel):
        ilsum = 0. + (tmpanchors.loc[
            ilabel.values == lbl, usrs] == lbl).sum(axis=1)
        anchors.loc[
            ilsum.index.tolist(),
            f'EM_inferred_label_count_{who}'] = ilsum

    # IMPORTANT! Get rid of "bogus" inferred labels. The EM
    # algorithm will assign a label even to nuclei with zero
    # observers (i.e. no SPs) based on prior probability. We
    # don't want that behavior.
    anchors.loc[
        anchors.loc[:, f'n_matches_{who}'] < 1,
        [f'EM_inferred_label_{who}',
         f'EM_inferred_label_count_{who}',
         f'EM_inferred_label_confidence_{who}',
         ]] = np.nan

    # IMPORTANT! Some anchors have such high discordance that
    # the inferred label does not "converge" and get assigned
    # a label not actually given by any participants. These
    # should be considered "unlabeled", even if none of the
    # participants called it so
    tochange = anchors.loc[
               :, f'EM_inferred_label_count_{who}'] < 1
    anchors.loc[tochange, f'EM_inferred_label_{who}'] = 'unlabeled'  # noqa
    anchors.loc[tochange, f'EM_inferred_label_count_{who}'] = \
        0. + (anchors.loc[tochange, usrs] == 'unlabeled').sum(
            axis=1)
    anchors.loc[
        tochange, f'EM_inferred_label_confidence_{who}'] = \
        ilprobs.loc[tochange, 'unlabeled']

    return anchors


def _add_EM_decision_on_boundary_correctness(
        anchors: DataFrame, who: str, initquality: float = 0.7,
        iterations: int = 70) -> DataFrame:
    """"""
    # convert to needed format
    usrs = ir.who[who]
    clicks = anchors.loc[:, [f'algorithmic_clicks_{who}'] + usrs].copy()
    for usr in ir.who[who]:
        clicks.loc[:, f'{usr}_click'] = clicks.loc[
            :, f'algorithmic_clicks_{who}'].apply(
            lambda x: 1 if usr in x else 0)
        clicks.loc[clicks.loc[
            :, f'{usr}'] == 'DidNotAnnotateFOV', f'{usr}_click'] = 2
        clicks.loc[clicks.loc[
            :, f'{usr}'] == 'undetected', f'{usr}_click'] = 2

    tmpanchors = clicks.loc[:, [f'{usr}_click' for usr in usrs]].copy()
    e2wl, w2el, label_set = gete2wlandw2el_fromdf(
        df=tmpanchors, missingval=2)

    # Run D&S EM algorithm
    em = EM(
        e2wl=e2wl, w2el=w2el, label_set=label_set,
        initquality=initquality)
    e2lpd, w2cm = em.Run(iterr=iterations)
    ilprobs = DataFrame.from_dict(e2lpd, orient='index')
    ilabel = Series(np.argmax(ilprobs.values, axis=1))
    ilabel = ilabel.map(
        {k: v for k, v in enumerate(list(ilprobs.columns))})
    anchors.loc[:, f'EM_decision_boundary_is_correct_{who}'] = ilabel

    return anchors


def add_all_EM_inferred_labels(dbcon):
    """Expectation-Maximization based method for ground truth inference
    from multi-observer datasets, as proposed by Dawid and Skene in 1979.

    Maximum Likelihood Estimation of Observer Error-Rates Using the EM Algorithm  # noqa
    Author(s): A. P. Dawid and A. M. Skene
    Source - Journal of the Royal Statistical Society. Series C (Applied
    Statistics), Vol. 28, No. 1 (1979), pp. 20-28
    Published by - Wiley for the Royal Statistical Society
    Stable URL - https://www.jstor.org/stable/2346806

    Parameters
    ----------
    dbcon

    Returns
    -------

    """
    for evalset in ir.EVALSET_NAMES:

        tablename = f'all_anchors_{evalset}'
        minious = read_sql_query(f"""
            SELECT DISTINCT "min_iou" from "{tablename}"
        ;""", dbcon).iloc[:, 0].tolist()

        # truth is inferred for each eval set, clustering min_iou,
        # and reference users independently
        for miniou in minious:

            # which groups of users do we consider as reference
            for who in ir.CONSENSUS_WHOS:

                print(
                    f'Adding EM label: {tablename}: min_iou=%.2f, {who}'
                    % miniou)

                wherestr = f"""
                        "min_iou" = {miniou}
                    AND "UNBIASED_n_matches_{who}" 
                        >= {ir.MIN_DETECTIONS_PER_ANCHOR}
                """
                anchors = read_sql_query(f"""
                    SELECT * FROM "{tablename}" WHERE {wherestr}
                ;""", dbcon)

                # add EM inferred label (detection and classification)
                anchors = _add_EM_for_detection_and_classification(
                    anchors=anchors, who=who)

                # add EM decision of whether algorithmic boundary is correct
                anchors = _add_EM_decision_on_boundary_correctness(
                    anchors=anchors, who=who)

                # delete old
                dbcon.execute(f"""
                    DELETE FROM "{tablename}" WHERE {wherestr} 
                ;""")

                # replace with new
                anchors.to_sql(
                    name=tablename, con=dbcon,
                    if_exists='append', index=False)


# %%===========================================================================


def add_unbiased_labels_to_db(dbcon):
    """Add labels from the U-control set to all conrresponding anchors.

    .. from the other evaluation sets. Also add a count of how many agree
    with the "true" label obtained from the unbiased set.

    """
    for evalset in ir.EVALSET_NAMES:

        tablename = f'all_anchors_{evalset}'

        # read relevant table
        print(f'Reading data: {tablename}')
        if evalset != 'U-control':
            anchors = read_sql_query(f"""
                SELECT * FROM "{tablename}"
            ;""", dbcon)
        else:
            anchors = None

        # which groups of users do we consider as reference
        for who in ir.CONSENSUS_WHOS:

            print(f'Adding unbiased label: {tablename}: {who}')

            for truthmethod in ['MV', 'EM']:

                extracol = f'{truthmethod}_inferred_label_confidence_{who}'

                if evalset == 'U-control':
                    col1 = f'{truthmethod}_inferred_label_{who}'
                    col2 = f'{truthmethod}_inferred_label_count_{who}'
                    query = f"""
                        UPDATE "{tablename}"
                        SET "UNBIASED_{col1}" = "{col1}",
                            "UNBIASED_{col2}" = "{col2}"
                    """
                    if truthmethod == 'EM':
                        query += f"""
                            , "UNBIASED_{extracol}" = "{extracol}"
                        """
                    dbcon.execute(f"""{query};""")
                    continue

                # read unbiased label
                col1 = f'{truthmethod}_inferred_label_{who}'
                select_part = f"""
                    SELECT "{col1}"
                """
                if truthmethod == 'EM':
                    select_part += f"""
                        , "{extracol}"
                    """
                tmpdf = read_sql_query(f"""
                    {select_part}
                    FROM "all_anchors_U-control"
                ;""", dbcon)
                colns = [col1]
                if truthmethod == 'EM':
                    colns.append(extracol)
                for coln in colns:
                    anchors.loc[:, f'UNBIASED_{coln}'] = tmpdf.loc[:, coln]

                # find out how many from THIS evalset agree with the
                # unbiased label
                ilabel = anchors.loc[:, f'UNBIASED_{col1}'].values
                uniq = {j for j in ilabel if j in ir.CLASSES}
                uccol = f'UNBIASED_{truthmethod}_inferred_label_count_{who}'  # noqa
                for lbl in uniq:
                    ilsum = 0. + (anchors.loc[
                        ilabel == lbl, ir.who[who]
                        ] == lbl).sum(axis=1)
                    anchors.loc[ilsum.index.tolist(), uccol] = ilsum
                anchors.loc[anchors.loc[
                    :, f'n_matches_{who}'] == 0, uccol] = np.nan

        # replace with new
        if anchors is not None:
            anchors.to_sql(
                name=tablename, con=dbcon, if_exists='replace',
                index=False)

# %%===========================================================================


def create_convenience_table_views(dbcon):
    """Create SQlite table views for convenient querying."""

    def _maybe_drop_view(name):
        try:
            dbcon.execute(f"""DROP VIEW "{name}";""")
        except:  # noqa
            pass

    def _separate_true_anchors(evalset, who, ubstr):
        """Separate anchors with 2+ matches."""
        print(f'_separate_true_anchors: {ubstr}: {evalset}: {who}')
        tablename = f'all_anchors_{evalset}'
        viewname = f'v1.1_true_anchors_{evalset}_{ubstr}{who}_AreTruth'
        _maybe_drop_view(viewname)
        dbcon.execute(f"""
            CREATE VIEW "{viewname}" AS
            SELECT * FROM "{tablename}"
            WHERE "{ubstr}n_matches_{who}" 
                >= {ir.MIN_DETECTIONS_PER_ANCHOR}
        ;""")

    def _exclude_suboptimal_anchors(evalset, who, ubstr):
        """Only keep anchors where EM inferred label is not "undetected"."""
        print(f'_exclude_suboptimal_anchors: {ubstr}: {evalset}: {who}')
        tablename = f'v1.1_true_anchors_{evalset}_{ubstr}{who}_AreTruth'
        viewname = f"{tablename.replace('v1.1_true', 'v2.1_consensus')}"
        _maybe_drop_view(viewname)
        dbcon.execute(f"""
            CREATE VIEW "{viewname}" AS
            SELECT * FROM "{tablename}"
            WHERE "{ubstr}EM_inferred_label_{who}" != "undetected"
        ;""")
        viewname = f"{tablename.replace('v1.1_true', 'v2.2_excluded')}"
        _maybe_drop_view(viewname)
        dbcon.execute(f"""
            CREATE VIEW "{viewname}" AS
            SELECT * FROM "{tablename}"
            WHERE "{ubstr}EM_inferred_label_{who}" = "undetected"
        ;""")

    def _get_final_anchors(evalset, who, ubstr):
        """Only keep consensus anchors at the desired clustering miniou."""
        print(f'_get_final_anchors: {ubstr}: {evalset}: {who}')
        tablename = f'v2.1_consensus_anchors_{evalset}_{ubstr}{who}_AreTruth'
        viewname = f"{tablename.replace('v2.1_consensus', 'v3.1_final')}"
        _maybe_drop_view(viewname)
        dbcon.execute(f"""
            CREATE VIEW "{viewname}" AS
            SELECT * FROM "{tablename}"
            WHERE "min_iou" = {ir.CMINIOU}
        ;""")

    for unbiased_is_truth in [True, False]:
        for evalset in ir.EVALSET_NAMES:
            for who in ir.CONSENSUS_WHOS:

                params = {
                    'ubstr': ir._ubstr(unbiased_is_truth),
                    'evalset': evalset,
                    'who': who,
                }

                # separate "true" anchors. i.e. detected by at least
                # x who on the unbiased control set
                _separate_true_anchors(**params)

                if any([
                    ir.TRUTHMETHOD != 'EM',
                    not ir.UNDETECTED_IS_A_CLASS
                ]):
                    raise NotImplementedError(
                        "The current implementation uses EM and 'undetected' "
                        "as a class, in order to exclude 'suboptimal' anchors!"
                    )

                # from the "true" anchors, only keep those that, using the
                # EM algorithm, were predicted to be "real"
                _exclude_suboptimal_anchors(**params)

                # for convenience, make a view for anchors at the optimum
                # clutering min_iou threshold
                _get_final_anchors(**params)


# %%===========================================================================


def main():

    # Where are the masks, contours, etc
    DATASETNAME = 'CURATED_v1_2020-03-29_EVAL'
    DATASETPATH = "/home/mtageld/Desktop/cTME/data/tcga-nucleus/"
    DATASETPATH = opj(DATASETPATH, DATASETNAME)

    # where to save stuff
    SAVEPATH = "/home/mtageld/Desktop/cTME/results/tcga-nucleus/interrater/"
    SAVEPATH = opj(SAVEPATH, DATASETNAME)
    _maybe_mkdir(SAVEPATH)
    _maybe_mkdir(opj(SAVEPATH, 'i1_anchors'))

    # get + save everyone's alias
    alias = ir.PARTICIPANT_ALIASES
    aliasdf = DataFrame.from_dict(alias, orient='index')
    aliasdf.to_csv(opj(SAVEPATH, 'i1_anchors', 'participant_aliases.csv'))

    # connect to sqlite database -- annotations
    db_path = opj(DATASETPATH, DATASETNAME + ".sqlite")
    sql_engine = create_engine('sqlite:///' + db_path, echo=False)
    dbcon_annots = sql_engine.connect()

    # to get FOV RGBs and visualize cluster medoids etc
    gc = CandygramAPI.connect_to_candygram()
    MPP = 0.2
    MAG = None

    # get information per evaluation set, user, and fov
    fovinfos = get_fovinfos_for_interrater(dbcon=dbcon_annots)
    with open(opj(SAVEPATH, 'i1_anchors', "fovinfos.json"), 'w') as f:
        json.dump(fovinfos, f, indent=4)

    # -------------------------------------------------------------------------

    for constrained in [True, False]:

        # connect to sqlite database -- anchors
        dbcon = _connect_to_anchor_db(SAVEPATH, constrained=constrained)

        # Get nucleus anchors, using pathologists (SP/JP) as truth
        # but also get the false anchors
        gana_kwargs = {
            'fovinfos': fovinfos,
            'get_medoids_kwargs': {
                'dbcon': dbcon_annots,  # annotations
                'who': 'All',
                'add_relative_bounds': True,
                'gc': gc, 'MPP': MPP, 'MAG': MAG,
                'constrained': constrained,
            },
            'dbcon': dbcon,  # anchors
            # 'min_ious': np.arange(0.125, 0.76, 0.125),
            'min_ious': [0.25, 0.5, 0.75],
            'fovs_to_use': None,
            'constrained': constrained,
        }
        get_all_nucleus_anchors_gtruth(**gana_kwargs)

        # Add Expectation-Maximization inferred labels
        add_all_EM_inferred_labels(dbcon=dbcon)

        # Add unbiased labels to all the eval sets
        add_unbiased_labels_to_db(dbcon=dbcon)

        # create convenience virtual tables
        create_convenience_table_views(dbcon=dbcon)


# %%===========================================================================

if __name__ == '__main__':
    main()
