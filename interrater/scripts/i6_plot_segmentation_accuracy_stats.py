from os.path import join as opj
from pandas import read_sql_query, concat
import matplotlib.pylab as plt
import seaborn as sns

from configs.nucleus_style_defaults import Interrater as ir
from interrater.interrater_utils import _maybe_mkdir, _connect_to_anchor_db


def _get_proportion_segmented(
        dbcon, unbiased_is_truth: bool, whoistruth: str, cls: str):
    """"""
    ubstr = 'UNBIASED_' if unbiased_is_truth else ''
    truthstr = f'{ubstr}{whoistruth}_AreTruth'
    anchors = []
    clstr = '' if cls == 'all' else \
        f'WHERE {ubstr}EM_inferred_label_{whoistruth} = "{cls}"'

    for evalset in ['B-control', 'E']:
        tmp = read_sql_query(f"""
            SELECT "fovname"
                 , COUNT(*) AS "n_anchors"
                 , AVG("EM_decision_boundary_is_correct_{whoistruth}") 
                    AS "avg_is_segmentation_{evalset}"
            FROM "v3.1_final_anchors_{evalset}_{truthstr}"
            {clstr}
            GROUP BY "fovname"
        ;""", dbcon)
        tmp.index = tmp.loc[:, 'fovname']
        keep = [f'avg_is_segmentation_{evalset}']
        if evalset == 'E':
            keep.append('n_anchors')
        anchors.append(tmp.loc[:, keep])
    anchors = concat(anchors, axis=1, join='inner')

    return anchors


def plot_proportion_segmented(
        savedir: str, dbcon, whoistruth: str, cls: str = 'all'):
    """"""
    # NOTE: In order to be able to plot this nicely, the total number
    #  of final anchors must be the same, so unbiased Ps must be the
    #  reference group
    unbiased_is_truth = True

    ubstr = 'UNBIASED_' if unbiased_is_truth else ''
    truthstr = f'{ubstr}{whoistruth}_AreTruth'
    where = opj(savedir, truthstr)
    _maybe_mkdir(where)
    _maybe_mkdir(opj(where, 'csv'))
    _maybe_mkdir(opj(where, 'plots'))

    # get the no of anchors and proportion segmented
    df = _get_proportion_segmented(
        dbcon=dbcon, unbiased_is_truth=unbiased_is_truth,
        whoistruth=whoistruth, cls=cls)

    # scatter, with size indicating number
    plt.figure(figsize=(5 * 1, 5.5 * 1))
    axis = sns.scatterplot(
        data=df,
        x=f'avg_is_segmentation_B-control',
        y=f'avg_is_segmentation_E',
        size='n_anchors', sizes=(10, 200), alpha=0.7, color='dimgray',
    )
    minn = -0.02
    maxn = 1.02
    axis.plot([0., maxn], [0., maxn], color='gray', linestyle='--')
    axis.set_xlim(minn, maxn)
    axis.set_ylim(minn, maxn)
    axis.set_xlabel('B-control', fontsize=11)
    axis.set_ylabel('E', fontsize=11)
    plt.title('N. segmentations / N. anchors', fontsize=14, fontweight='bold')
    plt.tight_layout(pad=0.3, w_pad=0.5, h_pad=0.3)
    savename = f'{truthstr}_evalset_proportionSegmented_comparison'
    plt.savefig(opj(where, 'plots', savename + '.svg'))
    plt.close()

    # save raw numbers
    df.to_csv(opj(where, 'csv', savename + '.csv'))


def _get_segmentation_accuracies_v1(
        dbcon, unbiased_is_truth: bool, whoistruth: str, metric: str):
    """"""
    truthstr = f'{"UNBIASED_" if unbiased_is_truth else ""}' \
               f'{whoistruth}_AreTruth'
    accuracy = []
    for evalset in ['B-control', 'E']:
        mcol = f'{metric}_{evalset}'
        tmp = read_sql_query(f"""
            SELECT "anchor_id", "algorithmic_vs_manual_{metric}" AS "{mcol}"
            FROM "v3.1_final_anchors_{evalset}_{truthstr}"
            WHERE "has_manual_boundary" = 1
              AND "EM_decision_boundary_is_correct_{whoistruth}" = 1
        ;""", dbcon)
        tmp.index = tmp.loc[:, 'anchor_id']
        accuracy.append(tmp.loc[:, [mcol]])

    accuracy = concat(accuracy, axis=1, join='inner')

    return accuracy


def plot_segmentation_accuracy_stats_v1(
        dbcon, savedir: str, unbiased_is_truth: bool, whoistruth: str):
    """"""
    truthstr = f'{"UNBIASED_" if unbiased_is_truth else ""}' \
               f'{whoistruth}_AreTruth'
    where = opj(savedir, truthstr)
    _maybe_mkdir(where)
    _maybe_mkdir(opj(where, 'csv'))
    _maybe_mkdir(opj(where, 'plots'))

    for metric in ['DICE', 'IOU']:

        df = _get_segmentation_accuracies_v1(
            dbcon=dbcon, unbiased_is_truth=unbiased_is_truth,
            whoistruth=whoistruth, metric=metric)

        # organize canvas and plot
        nperrow = 1
        nrows = 1
        plt.figure(figsize=(5 * nperrow, 5.5 * nrows))
        scprops = {'marker': 'o', 'alpha': 0.5, 's': 4 ** 2, 'color': 'dimgray'}
        # scatter plot with marginals and KDE density
        g = (
            sns.jointplot(
                x=f'{metric}_B-control', y=f'{metric}_E',
                data=df,  kind="scatter",
                xlim=(0.5, 1), ylim=(0.5, 1), **scprops
            ).plot_joint(
                sns.kdeplot, zorder=0, n_levels=10,
                cmap='YlOrRd', alpha=1.)
        )
        g.ax_joint.plot([0.5, 1.], [0.5, 1.], color='gray', linestyle='--')
        g.ax_joint.set_xlabel('B-control', fontsize=11)
        g.ax_joint.set_ylabel('E', fontsize=11)

        g.fig.suptitle(metric, fontsize=14, fontweight='bold')
        plt.tight_layout(pad=0.3, w_pad=0.5, h_pad=0.3)
        savename = f'{truthstr}_evalset_{metric}_comparison'
        plt.savefig(opj(where, 'plots', savename + '.svg'))
        plt.close()

        # save raw numbers
        df.to_csv(opj(where, 'csv', savename + '.csv'))


def _get_segmentation_accuracies_v2(
        dbcon, unbiased_is_truth: bool, whoistruth: str):
    """"""
    truthstr = f'{"UNBIASED_" if unbiased_is_truth else ""}' \
               f'{whoistruth}_AreTruth'
    accuracy = []
    for evalset in ['B-control', 'E']:
        tmp = read_sql_query(f"""
            SELECT "algorithmic_vs_manual_DICE" AS "DICE",
                   "algorithmic_vs_manual_IOU" AS "IOU",
                   "EM_decision_boundary_is_correct_{whoistruth}" AS "iscorrect"
            FROM "v3.1_final_anchors_{evalset}_{truthstr}"
            WHERE "has_manual_boundary" = 1
              AND "has_algorithmic_boundary" = 1
        ;""", dbcon)
        tmp.loc[:, 'evalset'] = evalset
        accuracy.append(tmp)

    accuracy = concat(accuracy, axis=0, ignore_index=True)

    return accuracy


def plot_segmentation_accuracy_stats_v2(
        dbcon, savedir: str, unbiased_is_truth: bool, whoistruth: str):
    """"""
    truthstr = f'{"UNBIASED_" if unbiased_is_truth else ""}' \
               f'{whoistruth}_AreTruth'
    where = opj(savedir, truthstr)
    _maybe_mkdir(where)
    _maybe_mkdir(opj(where, 'csv'))
    _maybe_mkdir(opj(where, 'plots'))

    # get df
    df = _get_segmentation_accuracies_v2(
        dbcon=dbcon, unbiased_is_truth=unbiased_is_truth,
        whoistruth=whoistruth)

    # organize canvas and plot
    nperrow = 2
    nrows = 1
    fig, ax = plt.subplots(nrows, nperrow, figsize=(5 * nperrow, 5.5 * nrows))

    for axno, metric in enumerate(['DICE', 'IOU']):

        axis = ax.ravel()[axno]

        # Draw a nested violinplot and split the violins for easier comparison
        axis = sns.violinplot(
            data=df, x='evalset', y=metric, hue='iscorrect',
            ax=axis, split=True, inner="quart", linewidth=1,
            palette={0: 'gold', 1: 'orangered'})

        axis.set_ylim(0., 1.)
        axis.set_title(metric, fontsize=14, fontweight='bold')
        axis.set_ylabel(metric, fontsize=11)

    plt.tight_layout(pad=0.3, w_pad=0.5, h_pad=0.3)
    savename = f'{truthstr}_evalset_violinplot_comparison'
    plt.savefig(opj(where, 'plots', savename + '.svg'))
    plt.close()

    # save raw numbers
    df.to_csv(opj(where, 'csv', savename + '.csv'))


def main():

    DATASETNAME = 'CURATED_v1_2020-03-29_EVAL'

    # where to save stuff
    BASEPATH = "/home/mtageld/Desktop/cTME/results/tcga-nucleus/interrater/"
    savedir = opj(BASEPATH, DATASETNAME, 'i6_SegmentationAccuracy')
    _maybe_mkdir(savedir)

    # connect to sqlite database -- anchors
    dbcon = _connect_to_anchor_db(opj(savedir, '..'))

    # Go through various evaluation sets & participant groups

    unbiased_is_truth = False

    for whoistruth in ir.CONSENSUS_WHOS:

        ubstr = "UNBIASED_" if unbiased_is_truth else ""
        print(f'{ubstr}{whoistruth}_AreTruth')

        # plot proportion of anchors that were agreed upon (by Ps) as
        # correctly segmented by the algorithm.
        # NOTE: Since the anchors here are paired, and the legend shows the
        #  no of FOVs per anchor, this by definition uses the unbiased control
        #  as a reference.
        plot_proportion_segmented(
            dbcon=dbcon, savedir=savedir, whoistruth=whoistruth)

        # compare accuracy stats for evalsets (coupled)
        plot_segmentation_accuracy_stats_v1(
            dbcon=dbcon, savedir=savedir,
            unbiased_is_truth=unbiased_is_truth, whoistruth=whoistruth)

        # compare accuracy stats for evalsets (independent)
        plot_segmentation_accuracy_stats_v2(
            dbcon=dbcon, savedir=savedir,
            unbiased_is_truth=unbiased_is_truth, whoistruth=whoistruth)


# %%===========================================================================

if __name__ == '__main__':
    main()
