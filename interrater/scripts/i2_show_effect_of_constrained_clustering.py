from os.path import join as opj
import sys
import numpy as np
from pandas import DataFrame, read_sql_query, concat
import matplotlib.pylab as plt

BASEPATH = "/home/mtageld/Desktop/cTME/"
sys.path.insert(0, BASEPATH)
from configs.nucleus_style_defaults import Interrater, DefaultAnnotationStyles
from interrater.interrater_utils import _get_rgb_for_interrater, \
    _visualize_bboxes_on_rgb, _get_color_from_rgbstr, _connect_to_anchor_db, \
    _maybe_mkdir

# NOTE: CandygramAPI is a private API that connects to a girder client
# The class contains private access tokens that cannot be shared publicly.
from configs.ACS_configs import CandygramAPI  # noqa


DefaultAnnotationStyles.STANDARD_STYLES['undetected'] = \
    DefaultAnnotationStyles.STANDARD_STYLES['unlabeled']


def vis_fov_anchors(
        axis, rgb, fovmeta, anchors, who='Ps', truthmethod='EM', title=''):
    """visualize medoids and labels by min iou for clustering."""
    # don't visualize anchors with < 3 observers
    # minm = 3
    # anchors = anchors.loc[anchors.loc[:, f'n_matches_{who}'] >= minm, :]

    # get FOV bounds -- shrink by a couple of pixels for aesthetics
    m = 3
    sf = fovmeta['sf']
    fovcont = {
        'xmin': sf * (fovmeta['maybe-xmin'] - fovmeta['XMIN']) + m,
        'xmax': sf * (fovmeta['maybe-xmax'] - fovmeta['XMIN']) - m,
        'ymin': sf * (fovmeta['maybe-ymin'] - fovmeta['YMIN']) + m,
        'ymax': sf * (fovmeta['maybe-ymax'] - fovmeta['YMIN']) - m,
        'color': 'yellow',
    }

    relative_anchors = anchors.loc[:, [
        'xmin_relative', 'ymin_relative', 'xmax_relative', 'ymax_relative']]
    relative_anchors.columns = [
        j.replace('_relative', '') for j in relative_anchors.columns]

    vis = _visualize_bboxes_on_rgb(
        rgb, xy_df=relative_anchors.copy(), fovcont=fovcont,
        totalcount=anchors.loc[:, f'n_matches_{who}'].values,
        # lblcount=anchors.loc[:, "best_label_count"].values,
        lblcolors=[
            _get_color_from_rgbstr(
                DefaultAnnotationStyles.STANDARD_STYLES[grp]
                ["lineColor"])
            if grp is not None else [0., 0., 0.]
            for grp in anchors.loc[:, f'{truthmethod}_inferred_label_{who}']
        ],
        bbox_linewidth=0.3,
        # bbox_linewidth=None,  # adaptive

        # determines thickness as count increases
        bbf=0.05 if who == 'Ps' else 0.02,

        # bbox_color='#525150',  # just detection
        bbox_color=None,  # detection + classificatino
    )
    axis.imshow(vis)
    axis.set_title(title, fontsize=12, fontweight='bold')


def _plot_n_anchors_by_constraint(axis, n_true_anchors, n_false_anchors):
    """Plot effect of clustering IOU threshold and dont-link constraint.

    Parameters
    ----------
    n_true_anchors : dict
        Has the keys "True" and "False" (constraint)
        Each entry is an np array where
        - first column is the min IOU threshold
        - second column is the number of resultant anchors (clusters)

    """
    legend = []

    for real in [True, False]:

        n_anchors = n_true_anchors if real else n_false_anchors

        for constrained in [False, True]:
            xy = np.array(n_anchors[constrained])
            axis.plot(
                xy[:, 0], xy[:, 1],
                linestyle='-' if real else '--',
                color='k' if constrained else 'gray',
                marker='.',
                linewidth=2,
            )
            prepend1 = "true" if real else "false"
            prepend2 = "" if constrained else "un"
            legend.append(f'{prepend1} anchors: {prepend2}constrained')

    axis.set_title("No of anchors", fontsize=12, fontweight='bold')
    axis.set_xlabel("Min IOU", fontsize=12)
    axis.set_ylabel("No of anchors", fontsize=12)
    axis.set_xlim(0, 0.8)
    axis.legend(legend)


def _get_and_plot_n_anchors_by_constraint(
        consensus_anchors, excluded_anchors, axis):
    """"""
    # get no of anchors for constrained vs unconstrained
    n_true_anchors = {k: [] for k in [True, False]}
    n_false_anchors = {k: [] for k in [True, False]}
    for real in [True, False]:
        for constrained in [True, False]:
            for min_iou in np.unique(
                    consensus_anchors[True].loc[:, 'min_iou']):
                anchors = consensus_anchors if real else excluded_anchors
                n_anchors = n_true_anchors if real else n_false_anchors
                # slice cluster medoids at this threshold
                keep = anchors[constrained].loc[:, 'min_iou'] == min_iou
                anchor_subset = anchors[constrained].loc[keep, :]
                n_anchors[constrained].append(
                    (min_iou, anchor_subset.shape[0]))

    # plot no of anchors
    _plot_n_anchors_by_constraint(
        axis=axis, n_true_anchors=n_true_anchors,
        n_false_anchors=n_false_anchors)


def _plot_anchor_detection_agreement_by_constraint(
        axis, consensus_anchors, n_didfov, who, miniou):
    """"""
    ta = consensus_anchors[True].copy()
    ta = ta.loc[ta.loc[:, "min_iou"] == miniou, f'n_matches_{who}'].values
    ta_un = consensus_anchors[False].copy()
    ta_un = ta_un.loc[ta_un.loc[
        :, "min_iou"] == miniou, f'n_matches_{who}'].values

    if (ta.shape[0] == 2) or (ta_un.shape[0] < 2):
        return

    mincnt = 0
    maxcnt = np.max([np.max(ta), np.max(ta_un)])
    bins = np.arange(mincnt, maxcnt + 1, 1)
    atleastn = [len(ta[ta >= n]) for n in bins]
    atleastn_un = [len(ta_un[ta_un >= n]) for n in bins]

    x = np.arange(len(bins))  # the label locations
    width = 0.35  # the width of the bars
    rects1 = axis.bar(
        x - width / 2, atleastn, width, label='constrained',
        edgecolor='k', facecolor='k')
    rects2 = axis.bar(
        x + width / 2, atleastn_un, width, label='unconstrained',
        edgecolor='k', facecolor='gray')

    upperlim = max(atleastn + atleastn_un) + 2
    rightlim = maxcnt - mincnt + 0.5
    axis.fill_betweenx(
        y=[0, upperlim], x1=n_didfov - mincnt + 0.5, x2=rightlim,
        color='gray', alpha=0.2,)

    # Add some text for labels, title and custom x-axis tick labels, etc.
    axis.set_ylim(0, upperlim)
    axis.set_xlim(-1, rightlim)
    axis.set_xlabel(f'Detected by at least x {who}', fontsize=12)
    axis.set_ylabel('No. of anchors', fontsize=12)
    axis.set_title(
        f'Anchor detection agreement (min_iou={miniou})',
        fontsize=12, fontweight='bold')
    axis.set_xticks(x)
    axis.set_xticklabels([int(j) for j in bins], rotation=45)
    axis.legend(loc='upper right')


def plot_effect_of_iouthresh_and_constraint(
        gc, dbcon, evalset, fovname, whoistruth='Ps', savename=None):
    """"""
    # get fov meta
    fovmeta = dict(read_sql_query(f"""
        SELECT *
        FROM "fov_meta"
        WHERE "fovname" = "{fovname}"
        ;""", dbcon[True]).iloc[0, :])

    # get no who annotated FOV
    n_didfov = len([
        j for j in fovmeta[f'participants_{evalset}'].split(',')
        if j in Interrater.who[whoistruth]])

    # get true and false anchors from database
    consensus_anchors = {
        cs: read_sql_query(f"""
            SELECT *
            FROM "v2.1_consensus_anchors_{evalset}_{whoistruth}_AreTruth"
            WHERE "fovname" = "{fovname}"
            ;""", dbcon[cs])
        for cs in [True, False]
    }

    # show all anchors, even if a single person detected them
    excluded_anchors = {
        cs: read_sql_query(f"""
                SELECT *
                FROM "v2.2_excluded_anchors_{evalset}_{whoistruth}_AreTruth"
                WHERE "fovname" = "{fovname}"
                ;""", dbcon[cs])
        for cs in [True, False]
    }
    for cs in [True, False]:
        consensus_anchors[cs] = concat(
            (consensus_anchors[cs], excluded_anchors[cs]), axis=0)

    # Init plot
    nrows = 1
    nperrow = 4
    fig, ax = plt.subplots(nrows, nperrow, figsize=(5 * nperrow, 5.5 * nrows))

    # # plot no of anchors
    # _get_and_plot_n_anchors_by_constraint(
    #     consensus_anchors=consensus_anchors,
    #     excluded_anchors=excluded_anchors, axis=ax[0])

    # plot anchor detection agreement
    _plot_anchor_detection_agreement_by_constraint(
        axis=ax[0], consensus_anchors=consensus_anchors, n_didfov=n_didfov,
        who=whoistruth, miniou=Interrater.CMINIOU)

    # calculate the scale factor & get RGB
    rgb = _get_rgb_for_interrater(
        gc=gc, bounds=fovmeta, slide_id=fovmeta['slide_id'])
    ax[1].imshow(rgb)

    # Visualize the true anchors
    for constrained in [True, False]:
        canchors = consensus_anchors[constrained].copy()
        canchors = canchors.loc[canchors.loc[
            :, 'min_iou'] == Interrater.CMINIOU, :]
        apstr = "" if constrained else "un"
        axno = 2 if constrained else 3
        vis_fov_anchors(
            axis=ax[axno], rgb=rgb, fovmeta=fovmeta,
            anchors=canchors.copy(), who=whoistruth,
            title=f"Anchors (min_iou={Interrater.CMINIOU}, "
                  f"{apstr}constrained)",
        )

    # now save everything
    plt.savefig(savename)
    plt.close()


def _get_fov_anchor_counts(dbcon, evalset, whoistruth):
    """get no of anchors."""
    anchor_counts = DataFrame()
    for constrained, dbc in dbcon.items():
        acnt = read_sql_query(f"""
            SELECT 
                "fovname", 
                count(*) as "n_anchors_{whoistruth}", 
                avg("n_matches_{whoistruth}") as "mean_n_matches_{whoistruth}"
            FROM "v3.1_final_anchors_{evalset}_{whoistruth}_AreTruth"
            GROUP BY "fovname"
        ;""", dbc)
        acnt.index = acnt.loc[:, 'fovname']
        acnt = acnt.iloc[:, 1:]
        if constrained:
            anchor_counts = acnt
        else:
            anchor_counts.loc[:, f'n_anchors_{whoistruth}_unconstrained'] = \
                acnt.loc[:, f'n_anchors_{whoistruth}']
            anchor_counts.loc[
                :, f'mean_n_matches_{whoistruth}_unconstrained'] = \
                acnt.loc[:, f'mean_n_matches_{whoistruth}']

    anchor_counts.loc[:, f'diff_n_anchors_{whoistruth}'] = np.abs(
        anchor_counts.loc[:, f'n_anchors_{whoistruth}']
        - anchor_counts.loc[:, f'n_anchors_{whoistruth}_unconstrained'])

    anchor_counts.loc[:, f'diff_mean_n_matches_{whoistruth}'] = np.abs(
        anchor_counts.loc[:, f'mean_n_matches_{whoistruth}']
        - anchor_counts.loc[:, f'mean_n_matches_{whoistruth}_unconstrained'])

    return anchor_counts


def run_constrained_clustering_by_fov_experiment(
        savepath, gc, whoistruth, evalset='U-control'):
    """Get medoids and vis constraining on a couple of fovs."""

    # connect to sqlite database -- anchors
    dbcon = {
        y: _connect_to_anchor_db(opj(savepath, '..'), constrained=y)
        for y in [True, False]
    }

    savedir = opj(savepath, f"{evalset}_{whoistruth}_AreTruth")
    _maybe_mkdir(savedir)

    # get fovs where there's a difference
    anchor_counts = _get_fov_anchor_counts(
        dbcon=dbcon, evalset=evalset, whoistruth=whoistruth)
    fovs_to_vis = list(anchor_counts.loc[anchor_counts.loc[
        :, f'diff_mean_n_matches_{whoistruth}'] > 0, :].index)

    for fovname in fovs_to_vis:

        print(f"visualizing {fovname}")

        # plot effect of clustering constraint
        plot_effect_of_iouthresh_and_constraint(
            gc=gc, dbcon=dbcon, fovname=fovname,
            whoistruth=whoistruth, evalset=evalset,
            savename=opj(savedir, f'constraintEffect_{fovname}.png'),
        )

# %%===========================================================================


def main():

    DATASETNAME = 'CURATED_v1_2020-03-29_EVAL'

    # where to save stuff
    BASEPATH = '/home/mtageld/Desktop/cTME/results/tcga-nucleus/interrater/'
    SAVEPATH = opj(BASEPATH, DATASETNAME, 'i2_ConstraintEffect')
    _maybe_mkdir(SAVEPATH)

    # to get FOV RGBs and visualize cluster medoids etc
    gc = CandygramAPI.connect_to_candygram()

    # check effect of constrained clustering on a couple of random FOVs
    for whoistruth in ['Ps']:  # Interrater.CONSENSUS_WHOS:
        for evalset in ['U-control', 'E']:
            run_constrained_clustering_by_fov_experiment(
                savepath=SAVEPATH, gc=gc, whoistruth=whoistruth,
                evalset=evalset)


# %%===========================================================================

if __name__ == '__main__':
    main()
