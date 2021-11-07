from os.path import join as opj
import numpy as np
from copy import deepcopy
from pandas import DataFrame, read_sql_query

from configs.nucleus_style_defaults import Interrater, \
    DefaultAnnotationStyles as das, NucleusCategories as ncg
from interrater.interrater_utils import _connect_to_anchor_db, remap_classes_in_anchorsdf
from interrater.krippendorff import alpha as kalpha


def _get_clmap_for_krippendorph(whats=None, clsgroup='main'):
    """
    We evaluate concordance for combined detection and classification
    (i.e. when a pathologist decides not to click a seed, we consider
    that to be a conscious decision of assigning an "undetected" class)
    versus classification alone (an unclicked seed is missing data).

    NOTE: We CANNOT evaluate detection alone because, by definition, we
    are removing seeds that are detected by < x observers, so the alpha
    values we get for detection alone are misleading; they're artifically
    low -- since we are SELECTIVELY removing nuclei with high detection
    concordance (people agree on them being "undetected").
    """
    assert clsgroup in ['raw', 'main', 'super']
    if whats is None:
        whats = ['detection_and_classification', 'classification']

    # dict mapping classes to an integer code
    if clsgroup == 'raw':
        dclasses = das.CLASSES
    elif clsgroup == 'main':
        dclasses = deepcopy(ncg.main_categs)
    else:
        dclasses = deepcopy(ncg.super_categs)

    clm = {dclasses[i]: i + 1 for i in range(len(dclasses))}
    dclasses.append('undetected')
    clm['undetected'] = max(clm.values()) + 1
    clmap = {what: deepcopy(clm) for what in whats}
    possible_values = {
        what: list({v for _, v in clmap[what].items()})
        for what in whats
    }
    clmap['classification']['undetected'] = np.nan
    clmap['detection_and_classification']['DidNotAnnotateFOV'] = np.nan
    clmap['classification']['DidNotAnnotateFOV'] = np.nan
    return clmap, possible_values


def _get_krippendorph_summary_by_detection_ease(
        dbcon, unbiased_is_truth, evalset, whoistruth, who,
        whichanchors, clsgroup, cumulative=True):
    """Get interrater agreement for subsets of anchors, sliced by the number
    of participants matched per anchor.
    """
    # init kappa agreement summary table
    whats = ['detection_and_classification', 'classification']
    summary_table = DataFrame(columns=[
        'class_grouping',
        'evalset', 'unbiased_is_truth', 'whoistruth', 'who',
        'whichanchors', 'min_iou', 'n_matches', 'n_anchors']
      + whats)

    # map classes to integer values (or nan)
    clmap, possible_values = _get_clmap_for_krippendorph(
        whats=whats, clsgroup=clsgroup)

    ubstr = Interrater._ubstr(unbiased_is_truth)
    tablename = f'{whichanchors}_anchors_{evalset}_' \
                f'{ubstr}{whoistruth}_AreTruth'

    minious = read_sql_query(f"""
        SELECT DISTINCT "min_iou" from "{tablename}"
    ;""", dbcon).iloc[:, 0].tolist()

    # get kappa agreement summary table
    for min_iou in minious:

        print(
            f'{clsgroup.upper()}: '
            f'Getting Krippendorph summary for {evalset}: {ubstr}{whoistruth}:'
            f' {who}: {whichanchors} anchors: min_iou = %.2f' % min_iou)

        unique_totals = read_sql_query(f"""
            SELECT DISTINCT "n_matches_{whoistruth}"
            FROM "{tablename}"
            WHERE "n_matches_{whoistruth}" NOT NULL
              AND "min_iou" = {min_iou} 
        ;""", con=dbcon).iloc[:, 0]

        for total in range(int(np.max(unique_totals.values))):

            # find seeds detected by (maybe AT LEAST) x people
            anchs_subset = read_sql_query(f"""
                SELECT * FROM "{tablename}"
                WHERE "min_iou" = {min_iou} 
                  AND "n_matches_{whoistruth}" 
                      {'>=' if cumulative else '='} {total} 
            ;""", dbcon)

            # remap classes to standard set
            anchs_subset = remap_classes_in_anchorsdf(
                anchors=anchs_subset, clsgroup=clsgroup,
                also_ilabel=True, remove_ambiguous=True,
                who_determines_ambig=whoistruth, how_ambig_is_determined='EM',
            )

            # restrict to users and transpose
            anchs_subset = anchs_subset.loc[:, Interrater.who[who]]
            anchs_subset = anchs_subset.T

            # init summary table
            idx = summary_table.shape[0]
            summary_table.loc[idx, 'class_grouping'] = clsgroup
            summary_table.loc[idx, 'evalset'] = evalset
            summary_table.loc[idx, 'unbiased_is_truth'] = unbiased_is_truth
            summary_table.loc[idx, 'whoistruth'] = whoistruth
            summary_table.loc[idx, 'who'] = who
            summary_table.loc[idx, 'whichanchors'] = whichanchors
            summary_table.loc[idx, 'min_iou'] = min_iou
            summary_table.loc[idx, 'n_matches'] = total
            summary_table.loc[idx, 'n_anchors'] = anchs_subset.shape[1]

            # get Krippendorph's alpha for detection and/or classification
            for what in whats:
                recoded_subset = anchs_subset.copy().applymap(
                    lambda x: clmap[what][x])
                summary_table.loc[idx, what] = kalpha(
                    reliability_data=recoded_subset,
                    value_domain=possible_values[what],
                    level_of_measurement='nominal')

    return summary_table


def save_krippendorph_summary(savepath, clsgroup):
    """"""
    assert clsgroup in ['raw', 'main', 'super']

    # connect to database
    dbcon = _connect_to_anchor_db(opj(savepath, '..'))

    # get and save krippendorph summary table
    for evalset in ['E', 'U-control']:

        # unbiased Ps, by definition, is U-control
        ubt = [False]
        if evalset != 'U-control':
            ubt.append(True)

        for unbiased_is_truth in ubt:
            for whoistruth in Interrater.CONSENSUS_WHOS:

                # unbiased NPs is not an interesting question
                if unbiased_is_truth and whoistruth == 'NPs':
                    continue

                for who in Interrater.CONSENSUS_WHOS:

                    # mixing Ps and NPs is meaningless
                    if who == 'All':
                        continue

                    # Ps compared to "truth" from NPs is not meaningful
                    if whoistruth == 'NPs' and who == 'Ps':
                        continue

                    # We only care about excluded anchors for main classes
                    # and for when Ps are truth, just to demonstrate that
                    # exclusion gets rid of bogus anchors
                    anchor_types = ['v2.1_consensus']
                    if (clsgroup == 'main') \
                            and (not unbiased_is_truth) \
                            and (whoistruth == 'Ps'):
                        anchor_types.append('v2.2_excluded')

                    for whichanchors in anchor_types:
                        summary = _get_krippendorph_summary_by_detection_ease(
                            dbcon, clsgroup=clsgroup,
                            unbiased_is_truth=unbiased_is_truth,
                            evalset=evalset,
                            whoistruth=whoistruth, who=who,
                            whichanchors=whichanchors)
                        summary.to_sql(
                            name=f'Krippendorph_byAnchorSubsets',
                            con=dbcon, if_exists='append', index=False)


# %%===========================================================================

def main():

    DATASETNAME = 'CURATED_v1_2020-03-29_EVAL'

    # where to save stuff
    BASEPATH = '/home/mtageld/Desktop/cTME/results/tcga-nucleus/interrater/'
    SAVEPATH = opj(BASEPATH, DATASETNAME, 'i1_anchors')

    # get krippendorph summary
    for clsgroup in ['main', 'super']:
        save_krippendorph_summary(savepath=SAVEPATH, clsgroup=clsgroup)


# %%===========================================================================

if __name__ == '__main__':
    main()
