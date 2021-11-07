from os.path import join as opj
import numpy as np
from pandas import DataFrame, concat
import matplotlib.pylab as plt
import seaborn as sns

from configs.nucleus_style_defaults import DefaultAnnotationStyles as das, Interrater
from interrater.interrater_utils import \
    _maybe_mkdir, get_fovs_annotated_by_almost_everyone, \
    _get_custom_uniform_cmap, _connect_to_anchor_db


def _get_detection_and_classification_tally(
        dbcon_anchors, unbiased_is_truth: bool, whoistruth: str,
        evalset: str, who: str):
    """"""
    # restrict to relevant FOV subset and anchors
    out = get_fovs_annotated_by_almost_everyone(
        dbcon_anchors=dbcon_anchors, unbiased_is_truth=unbiased_is_truth,
        whoistruth=whoistruth, evalset=evalset, who=who)

    obrange = list(range(out['maxn'] + 1))
    tallydfs = {
        cls: DataFrame(0., index=obrange[::-1], columns=obrange)
        for cls in ['all'] + das.CLASSES
    }
    truthcol = Interrater._get_truthcol(
        whoistruth=whoistruth, unbiased=unbiased_is_truth)

    # create a tally for each true label
    for cls in ['all'] + das.CLASSES:

        if cls == 'all':
            keep1 = [True] * out['anchors'].shape[0]
        else:
            keep1 = out['anchors'].loc[:, truthcol] == cls

        # go through anchors detected by nobs Ps/NPs
        for nobs in obrange:

            keep2 = out['anchors'].loc[:, f'n_matches_{who}'] == nobs
            keep = keep1 & keep2
            anchors = out['anchors'].loc[keep, Interrater.who[who]]

            if anchors.shape[0] == 0:
                continue

            is_correct = concat([
                out['anchors'].loc[keep, truthcol]] *
                len(Interrater.who[who]), axis=1)
            is_correct = 0 + (anchors.values == is_correct.values)

            # no of participants who got the correct label, per anchor
            ncorrect = np.sum(is_correct, axis=1)
            unq, cnt = np.unique(ncorrect, return_counts=True)
            for val, n in zip(unq, cnt):
                tallydfs[cls].loc[val, nobs] = n

    return tallydfs


def vis_detection_and_classification_tally(
        tallydfs, savename=None):
    """Plot agreement with "true" class (as determined by Ps on U-control).

    This is broken down by no of people who detected the seed.

    Parameters
    ----------
    tallydfs : dict

    savename : str
        path to save figure

    Returns
    -------
    None

    """
    # only keep classes observed
    # classes = [cls for cls, df in tallydfs.items() if np.sum(df.values)]
    classes = ['all'] + das.MAIN_CLASSES
    colors = das.COLORS
    colors['all'] = [0., 0., 0.]
    nperrow = 4
    nrows = int(np.ceil(len(classes) / nperrow))
    fig, ax = plt.subplots(nrows, nperrow, figsize=(5 * nperrow, 5.5 * nrows))
    clsno = 0
    for axis in ax.ravel():

        if clsno > len(classes) - 1:
            axis.plot([0, 1, 2], [0, 1, 2])
            axis.set_aspect('equal')
            continue

        cls = classes[clsno]
        clsno += 1

        X = DataFrame(tallydfs[cls].copy())
        # X.index = 1 + X.index[::-1]  # for plot to not have 0 at bottom
        # X.columns = 1 + X.columns

        sns.heatmap(
            X, ax=axis,
            mask=X == 0,
            vmin=0,  # 0.3
            # vmax=maxval,
            linewidths=.5,
            # annot=True, fmt='.0f',
            # cbar=False,
            cbar_kws={"shrink": 0.48},
            square=True,
            cmap=_get_custom_uniform_cmap(
                r=colors[cls][0], g=colors[cls][1], b=colors[cls][2], cmax=1),
        )

        # other props
        axis.set_title(cls, fontsize=16, fontweight='bold')
        if cls == 'all':
            xlab = 'No. who detected anchor'
            ylab = 'No. who agree with "true" label'
        else:
            xlab = 'No. who detected %s anchor' % cls
            ylab = 'No. who assigned label as %s' % cls
        axis.set_xlabel(xlab, fontsize=14)
        axis.set_ylabel(ylab, fontsize=14)

    plt.tight_layout(pad=0.3, w_pad=0.5, h_pad=0.3)

    if savename is None:
        plt.show()
    else:
        plt.savefig(savename)

    plt.close()


def get_and_plot_detection_and_classification_tally(
        savedir: str, unbiased_is_truth: bool, whoistruth: str,
        who: str, evalset: str):
    """Get a tally of detection and classification.

    For example, a tally dataframe for tumor nuclei, having a value of 43
    at row 3, column 5 means that there are 43 tumor nuclei (i.e. their REAL
    label is 'tumor') that were detected by 5 people, but only 3 of these
    people called it 'tumor'.
    """
    truthstr = f'{"UNBIASED_" if unbiased_is_truth else ""}{whoistruth}_AreTruth'  # noqa
    where = opj(savedir, truthstr)
    _maybe_mkdir(where)
    _maybe_mkdir(opj(where, 'csv'))
    _maybe_mkdir(opj(where, 'plots'))
    # connect to sqlite database -- anchors
    dbcon_anchors = _connect_to_anchor_db(opj(savedir, '..'))

    # get combined tally of detection and classification
    tallydfs = _get_detection_and_classification_tally(
        dbcon_anchors=dbcon_anchors, unbiased_is_truth=unbiased_is_truth,
        whoistruth=whoistruth, evalset=evalset, who=who)

    # save csvs
    prepend = f'{Interrater.TRUTHMETHOD}_{evalset}_{who}_{truthstr}'
    for cls, tallydf in tallydfs.items():
        tallydf.to_csv(opj(
            where, 'csv',
            f'{prepend}_{cls}_detection_and_classification_tally.csv'),
        )

    # now plot
    vis_detection_and_classification_tally(
        tallydfs=tallydfs,
        savename=opj(
            where, 'plots',
            f'{prepend}_detection_and_classification_tally.svg'),
    )


# %%===========================================================================


def main():

    DATASETNAME = 'CURATED_v1_2020-03-29_EVAL'

    # where to save stuff
    BASEPATH = "/home/mtageld/Desktop/cTME/results/tcga-nucleus/interrater/"
    SAVEPATH = opj(BASEPATH, DATASETNAME, 'i4_DetectionAndClassificationTally')
    _maybe_mkdir(SAVEPATH)

    # Go through various evaluation sets & participant groups
    for whoistruth in Interrater.CONSENSUS_WHOS:
        for unbiased_is_truth in [True, False]:
            for who in Interrater.CONSENSUS_WHOS:
                if (whoistruth == 'NPs') and (who == 'Ps'):
                    continue
                for evalset in Interrater.EVALSET_NAMES:
                    print(f'{whoistruth}IsTruth: {who}: {evalset}')
                    # now run main experiment
                    get_and_plot_detection_and_classification_tally(
                        savedir=SAVEPATH, unbiased_is_truth=unbiased_is_truth,
                        whoistruth=whoistruth, who=who, evalset=evalset)


# %%===========================================================================

if __name__ == '__main__':
    main()
