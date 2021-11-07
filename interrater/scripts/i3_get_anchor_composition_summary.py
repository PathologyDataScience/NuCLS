from os.path import join as opj
import numpy as np
from pandas import DataFrame
import matplotlib.pylab as plt
from matplotlib.colors import ListedColormap
from matplotlib.ticker import MaxNLocator

from configs.nucleus_style_defaults import Interrater, DefaultAnnotationStyles as das
from interrater.interrater_utils import _maybe_mkdir, _get_clmap, \
    get_fovs_annotated_by_almost_everyone, _connect_to_anchor_db, \
    remap_classes_in_anchorsdf
from configs.nucleus_model_configs import VisConfigs


def get_summary_counts_table(
        anchors, maxn, unbiased_is_truth, whoistruth, who,
        class_list=None, cumulative=True):
    """Get summary table of detection agreement, broken down by TRUE class.

    i.e. How many nuclei were detected by at least 6 people.

    Parameters
    ----------
    anchors: DataFrame
    maxn: int
        max possible no of observers. i.e no who annotated FOV.
    unbiased_is_truth: bool
    whoistruth: str
    who: str
    cumulative: bool
        i.e. which nucleus was detected by AT LEAST 6 observers, 5?, etc

    Returns
    -------
    DataFrame

    """
    # by definition, any true anchor is detected by 2+ {DETECTION_WHO} on the
    # U-control set, but this is not guaranteed for other eval sets, or
    # for SPs alone or NPs
    minn = 0

    class_list = class_list or das.CLASSES
    summary_table = DataFrame(columns=[f'n_{who}'] + class_list)
    for total in range(minn, maxn + 1):
        if cumulative:
            keep = anchors.loc[:, f'n_matches_{who}'] >= total
        else:
            keep = anchors.loc[:, f'n_matches_{who}'] == total
        keep = keep.values.tolist()
        subset = anchors.loc[keep, Interrater._get_truthcol(
            whoistruth=whoistruth, unbiased=unbiased_is_truth)]
        tally = {f'n_{who}': total, 'all': subset.shape[0]}
        tally.update({
            cls: subset[subset == cls].shape[0] for cls in class_list
        })
        summary_table = summary_table.append(tally, ignore_index=True)

    return summary_table


def _get_summary_percent_table(summary_counts_table, who, class_list=None):
    """Get summary table of detection agreement, broken down by TRUE class.

    Parameters
    ----------
    summary_counts_table: DataFrame
    who: str

    Returns
    -------
    DataFrame
    DataFrame

    """
    class_list = class_list or das.CLASSES
    # detection_composition -- starts at 100% (detected by at least the
    # mimumim no of observers) then tapers off
    detection_composition = summary_counts_table.copy()
    for cls in ['all'] + class_list:
        maxv = detection_composition.loc[:, cls].max()
        if maxv > 0:
            detection_composition.loc[:, cls] = \
                100 * detection_composition.loc[:, cls] / maxv
        else:
            detection_composition.loc[:, cls] = np.nan
    detection_composition.index = detection_composition.loc[:, f'n_{who}']

    # classification_composition adds to 100% vertically (per # detections)
    classification_composition = summary_counts_table.copy()
    for cls in class_list:
        classification_composition.loc[:, cls] = \
            100 * classification_composition.loc[:, cls] \
            / classification_composition.loc[:, 'all']
    classification_composition.index = classification_composition.loc[
        :, f'n_{who}']
    classification_composition = classification_composition.loc[:, class_list]

    return detection_composition, classification_composition


def vis_detection_tally_lineplot(
        summary_percent_table, axis, classes, xlab, ylab, who,
        title='', percent=True):
    """Visualize discordance versus no of people who detected anchor.

    Parameters
    ----------
    summary_percent_table : pandas DataFrame
        output from _get_summary_percent_table() or
        get_summary_counts_table()

    classes : list
        list of strings. Each entry is a "group" field corresponding to a
        unique class of points/annotations that is relevant to us.

    xlab : str
        x axis label

    ylab : str
        y axis label

    percent : bool
        are these percentages or row counts?

    who : str

    title : str

    Returns
    -------
    None

    """
    for cls in classes:
        props = {
            'marker': 'o',
            'linewidth': 3,
            'label': cls,
            'color': [j / 255. for j in VisConfigs.CATEG_COLORS[cls]],
            'alpha': 0.8,
        }
        axis.plot(
            summary_percent_table.loc[:, f"n_{who}"].values,
            summary_percent_table.loc[:, cls].values, **props)

    leftlim = 0
    rightlim = np.max(summary_percent_table.loc[:, f"n_{who}"].values)
    rightlim += 0.01 * rightlim
    if percent:
        upperlim = 101
    else:
        upperlim = np.max(summary_percent_table.loc[:, classes].values)
        upperlim += 0.01 * upperlim

    # force x ticks (no of observers) to be integers
    axis.xaxis.set_major_locator(MaxNLocator(integer=True))

    axis.set_ylim(0, upperlim)
    axis.set_xlim(leftlim, rightlim)
    axis.set_title(title, fontsize=14, fontweight='bold')
    axis.set_xlabel(xlab, fontsize=12)
    axis.set_ylabel(ylab, fontsize=12)


def vis_detection_tally_stacked_bar(
        summary_percent_table, axis, colormap, xlab, ylab, title=''):
    """Visualize discordance versus no of people who detected anchor.

    Parameters
    ----------
    summary_percent_table : pandas DataFrame
        output from _get_summary_percent_table() or
        get_summary_counts_table()

    xlab : str
        x axis label

    ylab : str
        y axis label

    Returns
    -------
    None

    """
    X = summary_percent_table.copy()
    ax = X.plot(
        ax=axis,
        kind='bar', stacked=True,
        colormap=colormap,
        legend=False,  # too cluttered
    )
    # force x ticks (no of observers) to be integers
    # axis.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_ylim(0, 100)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel(xlab, fontsize=12)
    ax.set_ylabel(ylab, fontsize=12)


def _plot_counts_summaries(
        cumulative_counts_table, detection_composition,
        Inferred_label_breakdown, who, class_list=None, savename=None):
    # visualize counts
    nrows = 1
    nperrow = 3
    fig, ax = plt.subplots(nrows, nperrow, figsize=(5 * nperrow, 5.5 * nrows))

    # rearranged classes so that main ones are overlayed
    class_list = class_list or das.CLASSES.copy()
    for axno, axis in enumerate(ax.ravel()):
        if axno == 0:
            vis_detection_tally_lineplot(
                cumulative_counts_table, axis=axis, who=who,
                classes=class_list.copy(),
                title='Raw counts',
                xlab=f"At least x {who} per anchor",
                ylab="No. of anchors",
                percent=False)
        elif axno == 1:
            vis_detection_tally_lineplot(
                detection_composition, axis=axis, who=who,
                classes=class_list.copy(),
                title='Ease of detection (%)',
                xlab=f"At least x {who} per anchor",
                ylab="Percentage of anchors (%)",
                percent=False)
            axis.set_ylim(0, 102)
            axis.legend(
                class_list, fontsize=8, title='"True" label',
                title_fontsize=10)
        elif axno == 2:
            vis_detection_tally_stacked_bar(
                Inferred_label_breakdown, axis=axis,
                colormap=ListedColormap([
                    [j / 255. for j in VisConfigs.CATEG_COLORS[cls]]
                    for cls in class_list
                ]),
                title=f'"True" label',  # noqa
                xlab=f"At least x {who} per anchor",
                ylab='"True" label breakdown (%)',
            )

    plt.tight_layout(pad=0.3, w_pad=0.5, h_pad=0.3)

    if savename is None:
        plt.show()
    else:
        plt.savefig(savename)
    plt.close()


def get_and_plot_all_summary_counts(
        savedir: str, unbiased_is_truth: bool, whoistruth: str,
        who: str, evalset: str, clsgroup: str):
    """"""
    assert clsgroup in ['raw', 'main', 'super']
    truthstr = f'{"UNBIASED_" if unbiased_is_truth else ""}{whoistruth}_AreTruth'  # noqa
    where = opj(savedir, truthstr)
    _maybe_mkdir(where)
    _maybe_mkdir(opj(where, 'csv'))
    _maybe_mkdir(opj(where, 'plots'))

    clmap, class_list = _get_clmap(clsgroup)
    class_list.remove('AMBIGUOUS')
    clmap['undetected'] = 'undetected'
    clmap['DidNotAnnotateFOV'] = 'DidNotAnnotateFOV'

    # connect to sqlite database -- anchors
    dbcon_anchors = _connect_to_anchor_db(opj(savedir, '..', '..'))

    # restrict to relevant FOV subset and anchors
    out = get_fovs_annotated_by_almost_everyone(
        dbcon_anchors=dbcon_anchors, unbiased_is_truth=unbiased_is_truth,
        whoistruth=whoistruth, evalset=evalset, who=who)

    # group classes as needed
    out['anchors'] = remap_classes_in_anchorsdf(
        anchors=out['anchors'], clsgroup=clsgroup)

    # Get tally of nuclei was detected by AT LEAST 6 observers, etc
    cumulative_counts_table = get_summary_counts_table(
        anchors=out['anchors'], maxn=out['maxn'],
        unbiased_is_truth=unbiased_is_truth, whoistruth=whoistruth,
        who=who, class_list=class_list)
    detection_composition, Inferred_label_breakdown = \
        _get_summary_percent_table(
            cumulative_counts_table, who=who, class_list=class_list)

    # save for reference
    prepend = f'{Interrater.TRUTHMETHOD}_{evalset}_{who}_{truthstr}'
    cumulative_counts_table.to_csv(opj(
        where, 'csv', f'{prepend}_counts_table.csv'))
    detection_composition.to_csv(opj(
        where, 'csv', f'{prepend}_detection_composition.csv'))
    Inferred_label_breakdown.to_csv(opj(
        where, 'csv', f'{prepend}_inferred_label_breakdown.csv'))

    # now plot
    _plot_counts_summaries(
        cumulative_counts_table=cumulative_counts_table,
        detection_composition=detection_composition,
        Inferred_label_breakdown=Inferred_label_breakdown,
        who=who, class_list=class_list,
        savename=opj(where, 'plots', f'{prepend}_count_summaries.svg'),
    )


# %%===========================================================================


def main():

    DATASETNAME = 'CURATED_v1_2020-03-29_EVAL'

    # where to save stuff
    BASEPATH = "/home/mtageld/Desktop/cTME/results/tcga-nucleus/interrater/"
    SAVEPATH = opj(BASEPATH, DATASETNAME, 'i3_AnchorSummary')
    _maybe_mkdir(SAVEPATH)

    VisConfigs.CATEG_COLORS['other_nucleus'] = [180] * 3

    # Go through various evaluation sets & participant groups
    for whoistruth in ['Ps']:  # Interrater.CONSENSUS_WHOS:
        for unbiased_is_truth in [False]:  # [True, False]
            for who in Interrater.CONSENSUS_WHOS:
                for evalset in ['E', 'U-control']:

                    if (whoistruth == 'NPs') and (who == 'Ps'):
                        continue

                    for clsgroup in ['main', 'super']:

                        print(
                            f'{clsgroup.upper()}: '
                            f'{whoistruth}IsTruth: {who}: {evalset}'
                        )

                        savedir = opj(SAVEPATH, clsgroup)
                        _maybe_mkdir(savedir)

                        # now run main experiment
                        get_and_plot_all_summary_counts(
                            savedir=savedir,
                            unbiased_is_truth=unbiased_is_truth,
                            whoistruth=whoistruth, who=who,
                            evalset=evalset, clsgroup=clsgroup)


# %%===========================================================================

if __name__ == '__main__':
    main()
