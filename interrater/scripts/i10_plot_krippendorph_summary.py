from os.path import join as opj
import numpy as np
from pandas import DataFrame, read_sql_query
import matplotlib.pylab as plt
from matplotlib import cm as mcmap
from matplotlib.ticker import MaxNLocator

from configs.nucleus_style_defaults import Interrater as ir
from interrater.interrater_utils import _connect_to_anchor_db, \
    _maybe_mkdir, _annotate_krippendorph_ranges


def _kripp_subplot(
        krippendorph_summary, axis, what, is_agreement, cutoff=None):
    """"""
    minx = np.min(krippendorph_summary.loc[:, f'n_matches'])
    maxx = np.max(krippendorph_summary.loc[:, f'n_matches'])
    if is_agreement:

        # annotate agreement ranges
        _annotate_krippendorph_ranges(
            axis=axis, minx=minx, maxx=maxx, shades=False)

    mx = 0
    for min_iou in [0.75, 0.5, 0.25]:
        ksummary = krippendorph_summary.loc[
           krippendorph_summary.loc[:, 'min_iou'] >= min_iou - 0.005, :]
        ksummary = ksummary.loc[ksummary.loc[
            :, 'min_iou'] < min_iou + 0.005, :]
        x = ksummary.loc[:, f'n_matches']
        if is_agreement:
            y = ksummary.loc[:, what]
        else:
            # y = 100 * ksummary.loc[:, what] / np.max(
            #     ksummary.loc[:, what].values)
            y = ksummary.loc[:, what]
            mx = np.max([np.max(y), mx])
        axis.plot(
            x, y,
            marker='o',
            # marker='o' if min_iou == miniou else '.',
            linestyle='-',
            # linestyle='-' if min_iou == miniou else '--',
            # linewidth=2. if min_iou == miniou else 1.,
            linewidth=1.5,
            c=mcmap.YlOrRd(min_iou + 0.2),
            # alpha=1. if min_iou == miniou else 0.7,
            label='min_iou=%.2f' % min_iou)

    axis.xaxis.set_major_locator(MaxNLocator(integer=True))
    if is_agreement:
        ymin = -0.02  # -0.22
        ymax = 1.02
    else:
        ymin = -mx * 0.02
        ymax = mx * 1.02
        # axis.set_ylim(0, 100)
        axis.set_ylim(ymin=0)
    minx = minx * 0.98
    maxx = maxx * 1.02
    if cutoff is not None:
        axis.fill_betweenx(
            y=[ymin, ymax], x1=minx, x2=cutoff - 0.2, color='gray', alpha=0.2)
    axis.set_ylim(ymin, ymax)
    axis.set_xlim(minx, maxx)
    axis.legend(fontsize=8)
    axis.set_title(what.replace('_', ' '), fontsize=14, fontweight='bold')
    axis.set_xlabel(
        f'At least x participants per anchor', fontsize=12)
    axis.set_ylabel(
        # 'Krippendorph Alpha' if is_agreement else '% anchors kept',
        'Krippendorph Alpha' if is_agreement else 'No. of anchors',
        fontsize=12)


def plot_krippendorph_figure(
        savedir: str, krippendorph_summary: DataFrame,
        unbiased_is_truth: bool, evalset: str, whoistruth: str,
        who: str, whichanchors: str):
    """"""
    path0 = opj(savedir, f'{whoistruth}AreTruth')
    path1 = opj(path0, f'{evalset}')
    for path in [path0, path1]:
        _maybe_mkdir(path)

    ubstr = ir._ubstr(unbiased_is_truth)
    print(
        f'Plotting Krippendorph for {evalset}: {ubstr}{whoistruth}AreTruth: '
        f'{who}: {whichanchors} anchors')
    ksummary = krippendorph_summary.loc[krippendorph_summary.loc[
        :, 'unbiased_is_truth'] == unbiased_is_truth, :]
    ksummary = ksummary.loc[
        ksummary.loc[:, 'evalset'] == evalset, :]
    ksummary = ksummary.loc[
        ksummary.loc[:, 'whoistruth'] == whoistruth, :]
    ksummary = ksummary.loc[ksummary.loc[:, 'who'] == who, :]
    ksummary = ksummary.loc[
        ksummary.loc[:, 'whichanchors'] == whichanchors, :]

    cutoff = ir.MIN_DETECTIONS_PER_ANCHOR if any([
        (evalset == 'U-control') and (who == whoistruth),
        (evalset != 'U-control') and (who == whoistruth) and (not unbiased_is_truth),  # noqa
    ]) else None

    if ksummary.shape[0] < 1:
        return

    nrows = 1
    nperrow = 3
    fig, ax = plt.subplots(nrows, nperrow, figsize=(
        5 * nperrow, 5.5 * nrows))
    for axno, axis in enumerate(ax.ravel()):
        if axno == 0:
            _kripp_subplot(
                axis=axis, what='n_anchors', is_agreement=False,
                cutoff=cutoff, krippendorph_summary=ksummary)
        elif axno == 1:
            _kripp_subplot(
                axis=axis, what='detection_and_classification', cutoff=cutoff,
                is_agreement=True, krippendorph_summary=ksummary)
        elif axno == 2:
            _kripp_subplot(
                axis=axis, what='classification', is_agreement=True,
                cutoff=cutoff, krippendorph_summary=ksummary)

    plt.tight_layout(pad=0.3, w_pad=0.5, h_pad=0.3)
    plt.savefig(opj(
        path1, f'krippendorph_{evalset}_{who}_{ubstr}{whoistruth}_AreTruth'
        f'_{whichanchors}.svg'))
    plt.close()


def plot_krippendorph_summary(savepath, clsgroup):
    """"""
    # connect to database
    dbcon = _connect_to_anchor_db(opj(savepath, '..'))

    # get krippendorph summary table
    krippendorph_summary = read_sql_query(f"""
        SELECT * FROM "Krippendorph_byAnchorSubsets"
        WHERE "class_grouping" = "{clsgroup}"
    ;""", dbcon)

    # now plot
    savedir = opj(savepath, '..', 'i10_Krippendorph', f'plots_{clsgroup}')
    _maybe_mkdir(savedir)
    _ = [
        plot_krippendorph_figure(
            savedir=savedir, krippendorph_summary=krippendorph_summary,
            unbiased_is_truth=unbiased_is_truth, evalset=evalset,
            whoistruth=whoistruth, who=who, whichanchors=whichanchors,
        )
        for evalset in ir.MAIN_EVALSET_NAMES
        for unbiased_is_truth in [True, False]
        for whoistruth in ir.CONSENSUS_WHOS
        for who in ir.CONSENSUS_WHOS
        for whichanchors in ['v2.1_consensus', 'v2.2_excluded']
    ]


def main():

    DATASETNAME = 'CURATED_v1_2020-03-29_EVAL'

    # where to save stuff
    BASEPATH = '/home/mtageld/Desktop/cTME/results/tcga-nucleus/interrater/'
    SAVEPATH = opj(BASEPATH, DATASETNAME, 'i1_anchors')

    kpath = opj(BASEPATH, DATASETNAME, 'i10_Krippendorph')
    _maybe_mkdir(kpath)

    # get krippendorph summary
    for clsgroup in ['main', 'super']:
        plot_krippendorph_summary(savepath=SAVEPATH, clsgroup=clsgroup)


# %%===========================================================================

if __name__ == '__main__':
    main()
