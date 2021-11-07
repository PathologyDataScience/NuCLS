from os.path import join as opj
import numpy as np
from pandas import read_sql_query, concat
import matplotlib.pylab as plt
import seaborn as sns

from configs.nucleus_style_defaults import Interrater as ir, NucleusCategories as ncg
from interrater.interrater_utils import _maybe_mkdir, \
    _connect_to_anchor_db, get_roc_and_auroc_for_who, \
    get_precision_recall_for_who


CLS = {
    'main': [j for j in ncg.main_categs if j != 'AMBIGUOUS'],
    'super': [j for j in ncg.super_categs if j != 'AMBIGUOUS'],
}


def _get_accuracy_stats(
        dbcon, whoistruth: str, unbiased_is_truth: bool,
        clsgroup: str, evalset=None):
    evquery = '' if evalset is None else f'AND "evalset" = "{evalset}"'
    # read accuracy stats
    classes = ['detection', 'classification'] + CLS[clsgroup]
    accuracy = read_sql_query(f"""
        SELECT *
        FROM "participant_AccuracyStats_{clsgroup}ClassGroup"
        WHERE "whoistruth" = "{whoistruth}"
          AND "unbiased_is_truth" = {0 + unbiased_is_truth}
          AND "total" > 1
          {evquery} 
          AND "participant" IN ({ir._get_sqlite_usrstr_for_who('All')})
          AND "class" IN ({ir._get_sqlitestr_for_list(classes)})
    ;""", dbcon)
    return classes, accuracy


def plot_participant_accuracy_stats(
        dbcon, savedir: str, unbiased_is_truth: bool, whoistruth: str,
        evalset: str, clsgroup: str):
    """"""
    truthstr = f'{"UNBIASED_" if unbiased_is_truth else ""}' \
               f'{whoistruth}_AreTruth'  # noqa
    where = opj(savedir, truthstr)
    _maybe_mkdir(where)
    _maybe_mkdir(opj(where, 'csv'))
    _maybe_mkdir(opj(where, 'plots'))

    classes, accuracy = _get_accuracy_stats(
        dbcon=dbcon, whoistruth=whoistruth, clsgroup=clsgroup,
        unbiased_is_truth=unbiased_is_truth, evalset=evalset)

    if whoistruth == 'Ps':
        tpr, fpr, roc_auc = get_roc_and_auroc_for_who(
            dbcon=dbcon, evalset=evalset, who='NPs', whoistruth=whoistruth,
            unbiased_is_truth=unbiased_is_truth, clsgroup=clsgroup)

    # to save raw values for calculating p-values later
    overalldf = []

    # organize canvas and plot
    nperrow = 4 if len(classes) <= 4 else 3
    nrows = int(np.ceil((len(classes)) / nperrow))
    fig, ax = plt.subplots(nrows, nperrow, figsize=(5 * nperrow, 5.5 * nrows))
    scprops = {'alpha': 0.75, 's': 9 ** 2, 'edgecolor': 'k'}
    axno = -1
    for axis in ax.ravel():
        axno += 1

        if axno == len(classes):
            break

        cls = classes[axno]
        isdetection = cls == 'detection'

        for who in ['NPs', 'JPs', 'SPs']:

            pstyle = ir.PARTICIPANT_STYLES[who]
            scprops.update({k: pstyle[k] for k in ['c', 'marker']})

            keep = accuracy.loc[:, 'participant'].apply(
                lambda x: x in ir.who[who])
            dfslice = accuracy.loc[keep, :]
            dfslice = dfslice.loc[dfslice.loc[:, 'class'] == cls, :]
            overalldf.append(dfslice)

            # add PR / ROC curve for inferred truth (from NPs)
            # versus the "actual" inferred truth (from SPs)
            if (whoistruth == 'Ps') and (who == 'NPs'):

                lprops = {'color': scprops['c'], 'alpha': 0.7, 'linewidth': 2}

                if isdetection:
                    # get precision-recalll curve
                    prc = get_precision_recall_for_who(
                        dbcon=dbcon, evalset=evalset, who='NPs',
                        whoistruth=whoistruth,
                        unbiased_is_truth=unbiased_is_truth)
                    # plot
                    axis.plot(
                        prc['recall'], prc['precision'], linestyle='-',
                        label=f'{who} "Truth" (AP=%0.2f)' % prc['AP'],
                        **lprops
                    )
                    axis.axhline(
                        prc['random'], xmin=0., xmax=1., c='gray',
                        linestyle='--', label='Random guess')
                elif cls == 'classification':
                    axis.plot(
                        fpr['micro'], tpr['micro'], linestyle='-',  # noqa
                        label=f'{who} "Truth" - MicroAvg (AUC=%0.2f)'
                              % roc_auc['micro'],  # noqa
                        **lprops
                    )
                    axis.plot(
                        fpr['macro'], tpr['macro'], linestyle='--',
                        label=f'{who} "Truth" - MacroAvg (AUC=%0.2f)'
                              % roc_auc['macro'],
                        **lprops
                    )
                else:
                    axis.plot(
                        fpr[cls], tpr[cls], linestyle='-',  # noqa
                        label=f'{who} "Truth" (AUC=%0.2f)' % roc_auc[cls],  # noqa
                        **lprops
                    )

            # scatter the various participants
            if isdetection:
                axis.scatter(
                    dfslice.loc[:, 'recall'], dfslice.loc[:, 'precision'],
                    label=f'{who}', **scprops)
            else:
                axis.scatter(
                    1 - dfslice.loc[:, 'specificity'],
                    dfslice.loc[:, 'sensitivity'],
                    label=f'{who}', **scprops)

        if isdetection:
            xlab, ylab = ('Recall (Sensitivity)', 'Precision (PPV)')
        else:
            axis.plot(
                [0., 0.5, 1.0], [0., 0.5, 1.0], c='gray',
                linestyle='--', label='Random guess')
            xlab, ylab = ('1 - Specificity (FPR)', 'Sensitivity (TPR)')

        axis.set_xlim(-0.02, 1.02)
        axis.set_ylim(-0.02, 1.02)
        axis.set_aspect('equal')
        axis.set_title(cls.capitalize(), fontsize=14, fontweight='bold')
        axis.set_xlabel(xlab, fontsize=11)
        axis.set_ylabel(ylab, fontsize=11)
        axis.legend(fontsize=8)

    # save plot
    plt.tight_layout(pad=0.3, w_pad=0.5, h_pad=0.3)
    savename = f'{truthstr}_{evalset}_accuracy_stats'
    plt.savefig(opj(where, 'plots', savename + '.svg'))
    plt.close()

    # save raw numbers
    overalldf = concat(overalldf, axis=0, ignore_index=True)
    overalldf.to_csv(opj(where, 'csv', savename + '.csv'))


def plot_participant_accuracy_stats_v2(
        dbcon, savedir: str, unbiased_is_truth: bool, whoistruth: str,
        clsgroup: str):
    """"""
    truthstr = f'{"UNBIASED_" if unbiased_is_truth else ""}' \
               f'{whoistruth}_AreTruth'
    where = opj(savedir, truthstr)
    _maybe_mkdir(where)
    _maybe_mkdir(opj(where, 'plots'))

    classes, accuracy = _get_accuracy_stats(
        dbcon=dbcon, whoistruth=whoistruth, clsgroup=clsgroup,
        unbiased_is_truth=unbiased_is_truth)

    # to save raw values for calculating p-values later
    overalldf = []

    # reorder evalsets
    tmp = []
    for evalset in ir.MAIN_EVALSET_NAMES:
        tmp.append(accuracy.loc[accuracy.loc[:, 'evalset'] == evalset, :])
    accuracy = concat(tmp, axis=0)

    # organize canvas and plot
    nperrow = 4 if len(classes) <= 4 else 3
    nrows = int(np.ceil((len(classes)) / nperrow))
    fig, ax = plt.subplots(nrows, nperrow, figsize=(5 * nperrow, 5.5 * nrows))
    scprops = {'alpha': 0.7, 's': 7 ** 2, 'edgecolor': 'k'}
    axno = -1
    for axis in ax.ravel():
        axno += 1

        if axno == len(classes):
            break

        cls = classes[axno]
        metric = 'F1' if cls == 'detection' else 'MCC'

        dfslice = accuracy.loc[accuracy.loc[:, 'class'] == cls, :].copy()
        dfslice.index = dfslice.loc[:, 'participant']
        dfslice.loc[:, 'who'] = 'NPs'
        for who in ['JPs', 'SPs']:
            for p in dfslice.index:
                if p in ir.who[who]:
                    dfslice.loc[p, 'who'] = who
        dfslice.loc[:, 'swho'] = dfslice.loc[:, 'who'].copy()
        dfslice.loc[dfslice.loc[:, 'swho'] == 'SPs', 'swho'] = 'Ps'
        dfslice.loc[dfslice.loc[:, 'swho'] == 'JPs', 'swho'] = 'Ps'
        dfslice = dfslice.loc[:, ['class', 'evalset', metric, 'who', 'swho']]
        overalldf.append(dfslice)

        # main boxplots
        bppr = {'alpha': 0.5}
        sns.boxplot(
            ax=axis, data=dfslice, x='evalset', y=metric, hue='swho',
            palette=[ir.PARTICIPANT_STYLES[who]['c'] for who in ['Ps', 'NPs']],
            boxprops=bppr, whiskerprops=bppr, capprops=bppr, medianprops=bppr,
            showfliers=False,
            # notch=True, bootstrap=5000,
        )

        # scatter each participant group
        for who in ['NPs', 'JPs', 'SPs']:

            pstyle = ir.PARTICIPANT_STYLES[who]
            scprops.update({k: pstyle[k] for k in ['c', 'marker']})
            plotme = dfslice.loc[dfslice.loc[:, 'who'] == who, :].copy()
            offset = -0.2 if who in ['JPs', 'SPs'] else 0.2
            plotme.loc[:, 'x'] = plotme.loc[:, 'evalset'].apply(
                lambda x: ir.MAIN_EVALSET_NAMES.index(x) + offset)
            plotme = np.array(plotme.loc[:, ['x', metric]])

            # add jitter
            plotme[:, 0] += 0.05 * np.random.randn(plotme.shape[0])

            # now scatter
            axis.scatter(
                plotme[:, 0], plotme[:, 1], label=f'{who}', **scprops)

        axis.set_ylim(0., 1.)
        # axis.set_ylim(0.5, 1.)
        axis.set_title(cls.capitalize(), fontsize=14, fontweight='bold')
        axis.set_ylabel(metric.capitalize(), fontsize=11)
        axis.legend()

    plt.tight_layout(pad=0.3, w_pad=0.5, h_pad=0.3)
    savename = f'{truthstr}_evalset_accuracy_comparison'
    plt.savefig(opj(where, 'plots', savename + '.svg'))
    plt.close()

    # save raw numbers
    overalldf = concat(overalldf, axis=0, ignore_index=True)
    overalldf.to_csv(opj(where, 'csv', savename + '.csv'))


def plot_participant_accuracy_stats_v3(
        dbcon, savedir: str, unbiased_is_truth: bool, whoistruth: str,
        clsgroup: str):
    """"""
    truthstr = f'{"UNBIASED_" if unbiased_is_truth else ""}' \
               f'{whoistruth}_AreTruth'  # noqa
    where = opj(savedir, truthstr)
    _maybe_mkdir(where)
    _maybe_mkdir(opj(where, 'csv'))
    _maybe_mkdir(opj(where, 'plots'))

    tpr = {}
    fpr = {}
    roc_auc = {}
    for evalset in ir.MAIN_EVALSET_NAMES:
        tpr[evalset], fpr[evalset], roc_auc[evalset] = \
            get_roc_and_auroc_for_who(
                dbcon=dbcon, evalset=evalset, who='NPs', clsgroup=clsgroup,
                whoistruth=whoistruth, unbiased_is_truth=unbiased_is_truth)

    # organize canvas and plot
    classes = ['detection', 'micro', 'macro'] + CLS[clsgroup]
    nperrow = 3
    nrows = int(np.ceil((len(classes)) / nperrow))
    fig, ax = plt.subplots(nrows, nperrow, figsize=(5 * nperrow, 5.5 * nrows))
    axno = -1
    for axis in ax.ravel():
        axno += 1

        if axno == len(classes):
            break

        cls = classes[axno]

        for evalset in ir.MAIN_EVALSET_NAMES:
            who = 'NPs'
            lprops = {
                'color': ir.PARTICIPANT_STYLES[who]['c'],
                'alpha': 1.,
                'linewidth': 2
            }
            lprops.update(ir.EVALSET_STYLES[evalset])

            if cls == 'detection':
                # get precision-recall curve
                prc = get_precision_recall_for_who(
                    dbcon=dbcon, evalset=evalset, who='NPs',
                    whoistruth=whoistruth,
                    unbiased_is_truth=unbiased_is_truth)
                # plot
                axis.plot(
                    prc['recall'], prc['precision'],
                    label=f'{evalset} (AP=%0.2f)' % prc['AP'],
                    **lprops
                )
                axis.axhline(
                    prc['random'], xmin=0., xmax=1., c='gray',
                    linestyle=lprops['linestyle'],
                    label=f'Random guess ({evalset})')
            else:
                axis.plot(
                    fpr[evalset][cls], tpr[evalset][cls],
                    label=f'{evalset} (AUC=%0.2f)' % roc_auc[evalset][cls],
                    **lprops
                )

        if cls == 'detection':
            xlab, ylab = ('Recall (Sensitivity)', 'Precision (PPV)')
        else:
            axis.plot(
                [0., 0.5, 1.0], [0., 0.5, 1.0], c='gray',
                linestyle='--', label='Random guess')
            xlab, ylab = ('1 - Specificity (FPR)', 'Sensitivity (TPR)')

        axis.set_xlim(-0.02, 1.02)
        axis.set_ylim(-0.02, 1.02)
        axis.set_aspect('equal')
        axis.set_title(cls.capitalize(), fontsize=14, fontweight='bold')
        axis.set_xlabel(xlab, fontsize=11)
        axis.set_ylabel(ylab, fontsize=11)
        axis.legend(fontsize=8)

    # save plot
    plt.tight_layout(pad=0.3, w_pad=0.5, h_pad=0.3)
    savename = f'{truthstr}_superimposed_auroc_curves'
    plt.savefig(opj(where, 'plots', savename + '.svg'))
    plt.close()


def main():

    DATASETNAME = 'CURATED_v1_2020-03-29_EVAL'

    # where to save stuff
    BASEPATH = "/home/mtageld/Desktop/cTME/results/tcga-nucleus/interrater/"
    SAVEDIR = opj(BASEPATH, DATASETNAME, 'i5_ParticipantAccuracy')
    _maybe_mkdir(SAVEDIR)

    # connect to sqlite database -- anchors
    dbcon = _connect_to_anchor_db(opj(SAVEDIR, '..'))

    # Go through various evaluation sets & participant groups
    for clsgroup in ['main', 'super']:

        savedir = opj(SAVEDIR, clsgroup)
        _maybe_mkdir(savedir)

        for whoistruth in ['Ps']:
            for unbiased_is_truth in [False]:

                ubstr = "UNBIASED_" if unbiased_is_truth else ""
                print(f'{clsgroup.upper()}: {ubstr}{whoistruth}_AreTruth')

                for evalset in ir.MAIN_EVALSET_NAMES:
                    # accuracy stats for a single avalset
                    plot_participant_accuracy_stats(
                        dbcon=dbcon, savedir=savedir,
                        unbiased_is_truth=unbiased_is_truth,
                        whoistruth=whoistruth, evalset=evalset,
                        clsgroup=clsgroup,
                    )

                # compare accuracy stats for various evalsets
                plot_participant_accuracy_stats_v2(
                    dbcon=dbcon, savedir=savedir,
                    unbiased_is_truth=unbiased_is_truth, whoistruth=whoistruth,
                    clsgroup=clsgroup,
                )

                # superimpose AUROC for various evalsets
                if whoistruth == 'Ps':
                    plot_participant_accuracy_stats_v3(
                        dbcon=dbcon, savedir=savedir,
                        unbiased_is_truth=unbiased_is_truth,
                        whoistruth=whoistruth, clsgroup=clsgroup,
                    )


# %%===========================================================================

if __name__ == '__main__':
    main()
