from os.path import join as opj
import sys
import numpy as np
from pandas import read_sql_query, concat
import matplotlib.pylab as plt
import seaborn as sns

from configs.nucleus_style_defaults import Interrater as ir
from interrater.interrater_utils import _maybe_mkdir, \
    _connect_to_anchor_db, _annotate_krippendorph_ranges


def plot_intrarater_stats(dbcon, savedir: str, clsgroup: str):
    """"""
    _maybe_mkdir(savedir)
    _maybe_mkdir(opj(savedir, 'csv'))
    _maybe_mkdir(opj(savedir, 'plots'))

    # read intrarater stats
    evalsets = ['B-control', 'E']
    stats = read_sql_query(f"""
        SELECT "participant", "second_evalset" AS "evalset"
             , "detection_and_classification", "classification"
             , "n_anchors_second_evalset" AS "n_anchors"
             , "n_clicks_second_evalset" AS "n_clicks"
        FROM "intra-rater_{clsgroup}ClassGroup"
        WHERE "participant" IN ({ir._get_sqlite_usrstr_for_who('All')})
          AND "first_evalset" = "U-control"
          AND "second_evalset" IN ({ir._get_sqlitestr_for_list(evalsets)})
    ;""", dbcon)
    stats.loc[:, 'psegmented'] = stats.loc[
        :, 'n_clicks'] / stats.loc[:, 'n_anchors']

    # to save raw values for calculating p-values later
    overalldf = []

    # reorder evalsets
    tmp = []
    for evalset in ir.EVALSET_NAMES:
        tmp.append(stats.loc[stats.loc[:, 'evalset'] == evalset, :])
    stats = concat(tmp, axis=0)

    # organize canvas and plot
    nperrow = 3
    nrows = 1
    fig, ax = plt.subplots(nrows, nperrow, figsize=(5 * nperrow, 5.5 * nrows))
    scprops = {'alpha': 0.7, 's': 7 ** 2, 'edgecolor': 'k'}
    axno = -1
    metrics = ['psegmented', 'detection_and_classification', 'classification']
    mdict = {
        'psegmented': 'N. segmentations / N. anchors',
        'detection_and_classification': "Kappa vs U-control (det. & classif.)",
        'classification': "Kappa vs U-control (classif.)",
    }
    for axis in ax.ravel():
        axno += 1
        metric = metrics[axno]

        dfslice = stats.copy()
        dfslice.index = dfslice.loc[:, 'participant']
        dfslice.loc[:, 'who'] = 'NPs'
        for who in ['JPs', 'SPs']:
            for p in dfslice.index:
                if p in ir.who[who]:
                    dfslice.loc[p, 'who'] = who
        dfslice.loc[:, 'swho'] = dfslice.loc[:, 'who'].copy()
        dfslice.loc[dfslice.loc[:, 'swho'] == 'SPs', 'swho'] = 'Ps'
        dfslice.loc[dfslice.loc[:, 'swho'] == 'JPs', 'swho'] = 'Ps'
        dfslice = dfslice.loc[:, ['evalset', metric, 'who', 'swho']]
        overalldf.append(dfslice)

        # annotate agreement ranges
        if axno > 0:
            _annotate_krippendorph_ranges(
                axis=axis, minx=0, maxx=2, shades=False)

        # main boxplots
        bppr = {'alpha': 0.5}
        sns.boxplot(
            ax=axis, data=dfslice, x='evalset', y=metric, hue='swho',
            palette=[ir.PARTICIPANT_STYLES[who]['c'] for who in ['Ps', 'NPs']],
            boxprops=bppr, whiskerprops=bppr, capprops=bppr, medianprops=bppr,
            showfliers=False, notch=False, bootstrap=5000)

        # scatter each participant group
        for who in ['NPs', 'JPs', 'SPs']:

            pstyle = ir.PARTICIPANT_STYLES[who]
            scprops.update({k: pstyle[k] for k in ['c', 'marker']})
            plotme = dfslice.loc[dfslice.loc[:, 'who'] == who, :].copy()
            offset = -0.2 if who in ['JPs', 'SPs'] else 0.2
            plotme.loc[:, 'x'] = plotme.loc[:, 'evalset'].apply(
                lambda x: evalsets.index(x) + offset)
            plotme = np.array(plotme.loc[:, ['x', metric]])

            # add jitter
            plotme[:, 0] += 0.05 * np.random.randn(plotme.shape[0])

            # now scatter
            axis.scatter(
                plotme[:, 0], plotme[:, 1], label=f'{who}', **scprops)

        axis.set_ylim(0., 1.)
        axis.set_title(mdict[metric], fontsize=14, fontweight='bold')
        axis.set_ylabel(metric.capitalize(), fontsize=11)
        axis.legend()

    plt.tight_layout(pad=0.3, w_pad=0.5, h_pad=0.3)
    savename = f'intra-rater_comparison'
    plt.savefig(opj(savedir, 'plots', savename + '.svg'))
    plt.close()

    # save raw numbers
    overalldf = concat(overalldf, axis=0)
    overalldf.to_csv(opj(savedir, 'csv', savename + '.csv'))

# %%===========================================================================


def main():

    DATASETNAME = 'CURATED_v1_2020-03-29_EVAL'

    # where to save stuff
    BASEPATH = "/home/mtageld/Desktop/cTME/results/tcga-nucleus/interrater/"
    SAVEDIR = opj(BASEPATH, DATASETNAME, 'i8_IntraRaterStats')
    _maybe_mkdir(SAVEDIR)

    # connect to sqlite database -- anchors
    dbcon = _connect_to_anchor_db(opj(SAVEDIR, '..'))

    # compare same participant on various evalsets
    for clg in ['main', 'super']:
        plot_intrarater_stats(
            dbcon=dbcon, savedir=opj(SAVEDIR, clg), clsgroup=clg)


# %%===========================================================================

if __name__ == '__main__':
    main()
