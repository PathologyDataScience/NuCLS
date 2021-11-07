from os.path import join as opj
import numpy as np
from pandas import DataFrame, read_sql_query, concat, Series
import matplotlib.pylab as plt
import seaborn as sns
from sklearn.manifold import MDS

from configs.nucleus_style_defaults import Interrater as ir, \
    DefaultAnnotationStyles as das  # noqa
from interrater.interrater_utils import _maybe_mkdir, \
    _connect_to_anchor_db, _annotate_krippendorph_ranges  # noqa


def _get_interrater(dbcon, evalsets: list, clsgroup: str, reorder=False):
    stats = read_sql_query(f"""
        SELECT * FROM "inter-rater_{clsgroup}ClassGroup"
        WHERE "first_participant" IN ({ir._get_sqlite_usrstr_for_who('All')})
          AND "second_participant" IN ({ir._get_sqlite_usrstr_for_who('All')})
          AND "evalset" IN (
            {ir._get_sqlitestr_for_list(evalsets)})
        --ORDER BY "first_participant", "second_participant"
    ;""", dbcon)
    tmp = stats.apply(
        lambda x: f'{x["first_participant"]}-{x["second_participant"]}',
        axis=1)
    stats.index = tmp

    # keep note of who belongs to what group
    for order in ('first', 'second'):
        stats.loc[:, f'{order}_who'] = \
            stats.loc[:, f'{order}_participant'].apply(
            lambda x: 'Ps' if x in ir.who['Ps'] else 'NPs')

    # get which pairs of groups are being compared
    stats.loc[:, 'comparison'] = stats.apply(
        lambda x: f'{x["first_who"]}-{x["second_who"]}', axis=1)

    if reorder:
        # reorder comparisons
        tmp = []
        for comp in ['Ps-Ps', 'Ps-NPs', 'NPs-NPs']:
            tmp.append(stats.loc[stats.loc[:, 'comparison'] == comp, :])
        stats = concat(tmp, axis=0)

        # reorder evalsets
        tmp = []
        for evalset in evalsets:
            tmp.append(stats.loc[stats.loc[:, 'evalset'] == evalset, :])
        stats = concat(tmp, axis=0)

    return stats


def _evalset_comparison_subplot(
        axis, stats: DataFrame, metric: str, evalsets: list):

    # annotate agreement ranges
    _annotate_krippendorph_ranges(axis=axis, minx=0, maxx=1.8, shades=False)

    # scatter each participant group
    scprops = {'alpha': 0.3, 's': 4 ** 2, 'edgecolor': 'gray', 'marker': 'o'}
    offsets = {'Ps-Ps': -0.3, 'Ps-NPs': 0, 'NPs-NPs': 0.3}
    other_scprops = {
        'Ps-Ps': {'c': ir.PARTICIPANT_STYLES['Ps']['c2']},
        'Ps-NPs': {'c': 'gray'},
        'NPs-NPs': {'c': ir.PARTICIPANT_STYLES['NPs']['c2']},
    }
    for comp in ['Ps-Ps', 'Ps-NPs', 'NPs-NPs']:
        plotme = stats.loc[stats.loc[:, 'comparison'] == comp, :].copy()
        offset = offsets[comp]
        plotme.loc[:, 'x'] = plotme.loc[:, 'evalset'].apply(
            lambda x: evalsets.index(x) + offset)
        plotme = np.array(plotme.loc[:, ['x', metric]])

        # add jitter
        plotme[:, 0] += 0.06 * np.random.randn(plotme.shape[0])

        # scatter
        scprops.update(other_scprops[comp])
        axis.scatter(plotme[:, 0], plotme[:, 1], **scprops)

    # main boxplot
    bppr = {'alpha': 0.5}
    palette = {
        'Ps-Ps': ir.PARTICIPANT_STYLES['Ps']['c'],
        'Ps-NPs': 'dimgray',
        'NPs-NPs': ir.PARTICIPANT_STYLES['NPs']['c'],
    }
    sns.boxplot(
        ax=axis, data=stats, x='evalset', y=metric, hue='comparison',
        palette=[v for _, v in palette.items()],
        boxprops=bppr, whiskerprops=bppr, capprops=bppr, medianprops=bppr,
        showfliers=False, notch=True, bootstrap=5000,
    )

    axis.set_ylim(0., 1.)
    axis.set_title(metric, fontsize=14, fontweight='bold')
    axis.set_ylabel("Cohen's Kappa", fontsize=11)
    axis.legend()


def plot_interrater_boxplots(dbcon, where: str, clsgroup: str):
    """"""
    _maybe_mkdir(opj(where, 'plots'))
    _maybe_mkdir(opj(where, 'csv'))
    # Note that for a fair comparison and to avoid confusion, we only include
    # the U-control and E set. This is because (almost) all participants
    # annotated both these setc, so we can later do a paired t-test (or
    # Wilcoxon), but only about half the participants did the B-control.
    # Besides, we're already "properly" compared the B-control when we did
    # the intra-rater stats
    evalsets = ['U-control', 'E']

    # read intrarater stats
    stats = _get_interrater(
        dbcon, evalsets=evalsets, clsgroup=clsgroup, reorder=True)

    # organize canvas and plot
    metrics = ['detection_and_classification', 'classification']
    nperrow = len(metrics)
    nrows = 1
    fig, ax = plt.subplots(nrows, nperrow, figsize=(5 * nperrow, 5.5 * nrows))
    axno = -1
    for axis in ax.ravel():
        axno += 1
        _evalset_comparison_subplot(
            axis=axis, stats=stats, metric=metrics[axno], evalsets=evalsets)

    savename = f'interrater_boxplots'
    plt.savefig(opj(where, 'plots', savename + '.svg'))
    plt.close()

    # raw numbers
    stats.to_csv(opj(where, 'csv', savename + '.csv'))


def _convert_stats_to_symmetric_matrix(stats: DataFrame) -> DataFrame:
    """convert stats into "proper" symmetric distance matrix"""
    stats = stats.loc[stats.index[::-1], :]
    X = stats.values
    X[X == 1.] = 0.
    np.fill_diagonal(X, 0.)
    X = X + X.T
    return DataFrame(X, index=stats.index, columns=stats.columns)


def _kappa_subplot(axis, stats: DataFrame, metric: str):

    # reorder from by concordance, then by experience
    order = []
    for exp in ['SP', 'JP', 'NP']:
        x = stats.loc[:, [j for j in stats.index if j.split('.')[0] == exp]]
        medians = Series(np.median(x.values, axis=0), index=x.columns)
        medians.sort_values(inplace=True, ascending=False)
        order.extend(list(medians.index))
    stats = stats.loc[order, order]

    # convert to diagonal for aesthetics
    tmp = np.tril(stats.values, 0)
    stats[tmp == 0] = 0.

    # make the diagonal from bottom left to top right for consistency
    # with other plots in the paper
    stats = stats.loc[:, stats.columns[::-1]]

    # main heatmap
    sns.heatmap(
        stats, ax=axis, linewidths=.5,
        mask=stats.values < 0.001,
        vmin=0, vmax=1,
        # cbar=False,
        cbar_kws={"shrink": 0.48},
        square=True,
        cmap='YlOrRd',  # 'binary'
    )
    # other props
    axis.set_title(metric.replace('_', ' '), fontsize=14, fontweight='bold')
    axis.set_xlabel('Participant 2', fontsize=11)
    axis.set_ylabel('Participant 1', fontsize=11)


def _mds_subplot(axis, X: DataFrame, metric: str):
    """"""
    # calculate MDS matrix
    mds_model = MDS(
        n_components=2, max_iter=100, n_init=1,
        dissimilarity='precomputed')
    mds = mds_model.fit_transform(1 - X.values)  # distance is 1 - kappa
    mds = DataFrame(mds, index=X.index)

    # scatter by experience
    scprops = {'alpha': 0.9, 's': 9 ** 2}
    for exp in ['NPs', 'JPs', 'SPs']:
        tmpstyle = ir.PARTICIPANT_STYLES[exp]
        scprops.update({
            'label': exp,
            'marker': tmpstyle['marker'],
            'c': tmpstyle['c'],
            'edgecolor': 'k',
        })
        keep = [j for j in mds.index if j.split('.')[0] == exp[:-1]]
        axis.scatter(mds.loc[keep, 0], mds.loc[keep, 1], **scprops)

        # scatter participant names
        scatter_labels = list(mds.index)
        jitter = 0.04 * np.abs(
            (np.max(mds.values[:, 0]) - np.min(mds.values[:, 0])))
        for y in range(mds.shape[0]):
            axis.annotate(
                scatter_labels[y],
                xy=(mds.iloc[y, 0], mds.iloc[y, 1] + jitter),
                horizontalalignment='center', verticalalignment='center',
                fontsize=8)

    # other props
    minval = np.min(mds.values) - 0.1
    maxval = np.max(mds.values) + 0.1
    axis.set_xlim(minval, maxval)
    axis.set_ylim(minval, maxval)
    axis.set_aspect('equal')
    axis.set_title(metric.replace('_', ' '), fontsize=14, fontweight='bold')
    axis.set_xlabel('Participant 2', fontsize=11)
    axis.set_ylabel('Participant 1', fontsize=11)
    axis.legend()


def plot_interrater_pairs(dbcon, where: str, evalset: str, clsgroup: str):

    # read intrarater stats
    tmpstats = _get_interrater(dbcon, evalsets=[evalset], clsgroup=clsgroup)

    # convert to required format & save raw
    metrics = ['detection_and_classification', 'classification']
    ps = np.unique(tmpstats.loc[
       :, ['first_participant', 'second_participant']].values).tolist()
    ps = [j for j in ir.All if j in ps]
    all_stats = {
        mt: DataFrame(0., index=ps[::-1], columns=ps) for mt in metrics}
    for _, row in tmpstats.iterrows():
        for mt in metrics:
            all_stats[mt].loc[
                row['first_participant'],
                row['second_participant']
            ] = row[mt]

    # organize canvas and plot
    savename = f'interrater_pairs_{evalset}'
    nperrow = 2 * len(metrics)
    nrows = 1
    fig, ax = plt.subplots(nrows, nperrow, figsize=(7 * nperrow, 5.5 * nrows))
    axno = 0
    for metric in metrics:

        stats = _convert_stats_to_symmetric_matrix(all_stats[metric])

        # save raw numbers
        stats.to_csv(opj(where, 'csv', f'{savename}_{metric}.csv'))

        # pairwise cohen's kappa
        _kappa_subplot(
            axis=ax.ravel()[axno], stats=stats.copy(), metric=metric)
        axno += 1

        # MDS reduced dimensionality version of the pairwise matrix
        _mds_subplot(axis=ax.ravel()[axno], X=stats, metric=metric)
        axno += 1

    plt.savefig(opj(where, 'plots', savename + '.svg'))
    plt.close()


def main():

    DATASETNAME = 'CURATED_v1_2020-03-29_EVAL'

    # where to save stuff
    BASEPATH = "/home/mtageld/Desktop/cTME/results/tcga-nucleus/interrater/"
    savedir = opj(BASEPATH, DATASETNAME, 'i9_InterRaterStats')
    _maybe_mkdir(savedir)

    # connect to sqlite database -- anchors
    dbcon = _connect_to_anchor_db(opj(savedir, '..'))

    # plot kappa matrix and MDS plot
    for clg in ['main', 'super']:

        where = opj(savedir, clg)
        _maybe_mkdir(where)

        # compare various evalsets in terms of inter-rater concordance
        plot_interrater_boxplots(dbcon=dbcon, where=where, clsgroup=clg)

        for evalset in ir.MAIN_EVALSET_NAMES:
            plot_interrater_pairs(
                dbcon=dbcon, where=where, evalset=evalset, clsgroup=clg)


# %%===========================================================================

if __name__ == '__main__':
    main()
