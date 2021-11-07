from os.path import join as opj
import numpy as np
from pandas import DataFrame, read_sql_query
import matplotlib.pylab as plt
import seaborn as sns

from configs.nucleus_style_defaults import Interrater as ir, NucleusCategories as ncg
from interrater.interrater_utils import _maybe_mkdir, _connect_to_anchor_db


def _get_confmat_per_participant(
        dbcon, unbiased_is_truth: bool, whoistruth: str, who: str,
        evalset: str, discard_unmatched: bool, clsgroup: str, minn=10):
    """Get normalized confusion matrix per pathologist .. i.e. what % of
    annotations labeled as tumor by the pathologist actually belong
    to a "true" tumor nucleus?
    """
    class_list = ncg.main_categs if clsgroup == 'main' else ncg.super_categs
    # read confusion per pathologist
    clssumstr = ir._get_sqlitestr_for_list(
        what=class_list[::-1], prefix='SUM(', postfix=')')
    cperp = read_sql_query(f"""
        SELECT "EM_truth", {clssumstr}
        FROM "confusion_byParticipant_{clsgroup}ClassGroup"
        WHERE "unbiased_is_truth" = {int(unbiased_is_truth)}
          AND "whoistruth" = "{whoistruth}"
          AND "evalset" = "{evalset}"
          AND "participant" IN ({ir._get_sqlite_usrstr_for_who(who)})
        GROUP BY "EM_truth"
    ;""", dbcon)
    cperp.index = cperp.loc[:, 'EM_truth']
    cperp.drop('EM_truth', axis=1, inplace=True)
    cperp.columns = [j.split('SUM("')[1].split('")')[0] for j in cperp.columns]

    # reorder
    if discard_unmatched:
        cperp = cperp.loc[class_list, :]
    else:
        cperp = cperp.loc[class_list + ['non-existent'], :]

    # make sure the label truth is on x axis
    cperp = cperp.T

    # # ignore classes that are not in truth AND not annotated by anyone
    # for cls in ir.CLASSES:
    #     rsum = np.sum(cperp.loc[cls, :].values)
    #     csum = np.sum(cperp.loc[:, cls].values)
    #     if (rsum + csum) == 0:
    #         cperp.drop(cls, axis=0, inplace=True)
    #         cperp.drop(cls, axis=1, inplace=True)

    # Normalize.
    # Note that percentage is deceptive when total is small, so we threshold
    for cls in cperp.index:
        total = np.sum(cperp.loc[cls, :].values)
        if total >= minn:
            cperp.loc[cls, :] = 100 * cperp.loc[cls, :] / total
        else:
            cperp.loc[cls, :] = 0.

    return cperp


def _get_confmat_per_anchor(
        dbcon, unbiased_is_truth: bool, whoistruth: str, who: str,
        evalset: str, clsgroup: str) -> DataFrame:
    """Get normalized confusion matrix showing on average, out of the
    n{who} observers how many picked, say "tumor" for a true anchor (i.e. a
    "real" nucleus) whose true label is "tumor".
    """
    class_list = ncg.main_categs if clsgroup == 'main' else ncg.super_categs
    # read confusion per pathologist
    clssumstr = ir._get_sqlitestr_for_list(
        what=['undetected'] + class_list[::-1], prefix='SUM(', postfix=')')
    cpera = read_sql_query(f"""
        SELECT "n_who_AnnotatedFOV", "EM_truth", {clssumstr}
        FROM "confusion_byAnchor_{clsgroup}ClassGroup"
        WHERE "unbiased_is_truth" = {int(unbiased_is_truth)}
          AND "whoistruth" = "{whoistruth}"
          AND "evalset" = "{evalset}"
          AND "who" = "{who}"
        GROUP BY "EM_truth"
    ;""", dbcon)
    maxn = int(cpera.loc[0, 'N_who_AnnotatedFOV'])
    cpera.index = cpera.loc[:, 'EM_truth']
    cpera.drop(['EM_truth', 'N_who_AnnotatedFOV'], axis=1, inplace=True)
    cpera.columns = [j.split('SUM("')[1].split('")')[0] for j in cpera.columns]

    # reorder
    cpera = cpera.loc[class_list, :]

    # make sure the label truth is on x axis
    cpera = cpera.T

    # # ignore classes that are not in truth AND not annotated by anyone
    # for cls in ir.CLASSES:
    #     rsum = np.sum(cpera.loc[cls, :].values)
    #     csum = np.sum(cpera.loc[:, cls].values)
    #     if (rsum + csum) == 0:
    #         cpera.drop(cls, axis=0, inplace=True)
    #         cpera.drop(cls, axis=1, inplace=True)

    return cpera, maxn


def _plot_confusion(
        confmat_by_anchor: DataFrame, confmat_by_participant: DataFrame,
        n_participants: int, discard_unmatched: bool, savename: str) -> None:
    # Add extra row/column to make square for aesthetics
    idxs = list(confmat_by_participant.index)
    confmat_by_participant.loc['', :] = 0.
    confmat_by_participant = confmat_by_participant.loc[[''] + idxs, :]
    if not discard_unmatched:
        confmat_by_anchor.loc[:, ''] = 0.

    # organize canvas and plot
    nperrow = 2
    nrows = 1
    fig, ax = plt.subplots(nrows, nperrow, figsize=(7 * nperrow, 8 * nrows))
    axno = -1
    for axis in ax.ravel():

        axno += 1
        perpathol = axno == 0

        if perpathol:
            X = confmat_by_participant.copy()
            X.rename(columns={'non-existent': 'non-existent (FP)'})
        else:
            X = confmat_by_anchor.copy()
            X.rename(index={'undetected': 'undetected (FN)'})

        X = DataFrame(np.float32(X), index=X.index, columns=X.columns)
        sns.heatmap(
            X, ax=axis,
            annot=True, annot_kws={'fontsize': 8},
            fmt='.0f',
            mask=X < 0.5,
            # mask=X < 1 if perpathol else X < 0.1,
            # fmt='.0f' if perpathol else '.1f',
            vmin=0, vmax=100 if perpathol else n_participants,
            linewidths=.5,
            # cbar=idx == 0,
            cbar=False,
            square=True,
            cmap='YlOrRd',
            # cmap=_get_custom_uniform_cmap(r=80, g=80, b=80),
        )

        # other props
        mstr = 'matched' if discard_unmatched else ''
        perpathol_title = \
            f'"What % of {mstr} anchors placed by the participant and ' \
            'labeled as\ntumor actually correspond to a tumor nucleus?"'
        perseed_title = \
            f'"How many of our {n_participants} participants detected the '\
            '"typical"\ntumor nucleus? And what label did they assign?"\n'
        axis.set_title(
            perpathol_title if perpathol else perseed_title,
            fontsize=9, fontstyle='italic')
        xlab = '"True" label'
        ylab = 'Participant label'
        axis.set_xlabel(xlab, fontsize=11)
        axis.set_ylabel(ylab, fontsize=11)

    plt.tight_layout(pad=0.3, w_pad=0.5, h_pad=0.3)

    if savename is None:
        plt.show()
    else:
        plt.savefig(savename)

    plt.close()


def plot_participant_confusions(
        dbcon, savedir: str, unbiased_is_truth: bool, whoistruth: str,
        who: str, evalset: str, clsgroup: str, discard_unmatched: bool = True):
    """"""
    truthstr = f'{"UNBIASED_" if unbiased_is_truth else ""}' \
               f'{whoistruth}_AreTruth'
    where = opj(savedir, truthstr)
    _maybe_mkdir(where)
    _maybe_mkdir(opj(where, 'csv'))
    _maybe_mkdir(opj(where, 'plots'))

    # get confusion matrices by participant and seed
    params = {
        'dbcon': dbcon,
        'unbiased_is_truth': unbiased_is_truth,
        'whoistruth': whoistruth,
        'who': who,
        'evalset': evalset,
        'clsgroup': clsgroup,
    }
    confusions = {
        'per_participant': _get_confmat_per_participant(
            discard_unmatched=discard_unmatched, **params)
    }
    confusions['per_anchor'], n_who = _get_confmat_per_anchor(**params)

    # save csv
    savename = f'{who}_{truthstr}_{evalset}_confusions'
    for cs, confmat in confusions.items():
        confmat.to_csv(opj(where, 'csv', savename + '.csv'))

    # Now plot both conusion matrices
    _plot_confusion(
        confmat_by_anchor=confusions['per_anchor'],
        confmat_by_participant=confusions['per_participant'],
        n_participants=n_who, discard_unmatched=discard_unmatched,
        savename=opj(where, 'plots', savename + '.svg'),
    )


def main():

    DATASETNAME = 'CURATED_v1_2020-03-29_EVAL'

    # where to save stuff
    BASEPATH = "/home/mtageld/Desktop/cTME/results/tcga-nucleus/interrater/"
    SAVEDIR = opj(BASEPATH, DATASETNAME, 'i7_ParicipantConfusions')
    _maybe_mkdir(SAVEDIR)

    # connect to sqlite database -- anchors
    dbcon = _connect_to_anchor_db(opj(SAVEDIR, '..'))

    # Go through various evaluation sets & participant groups
    for clsgroup in ['main', 'super']:

        savedir = opj(SAVEDIR, clsgroup)
        _maybe_mkdir(savedir)

        for whoistruth in ['Ps']:  # ir.CONSENSUS_WHOS:
            for unbiased_is_truth in [False]:  # [True, False]
                for who in ir.CONSENSUS_WHOS:
                    if (whoistruth == 'NPs') and (who == 'Ps'):
                        continue
                    for evalset in ['E', 'U-control']:  # ir.MAIN_EVALSET_NAMES

                        ubstr = "UNBIASED_" if unbiased_is_truth else ""
                        print(
                            f'{clsgroup.upper()}: '
                            f'{ubstr}{whoistruth}_AreTruth: {who}: {evalset}'
                        )

                        # compare accuracy stats for various evalsets
                        plot_participant_confusions(
                            dbcon=dbcon, savedir=savedir,
                            unbiased_is_truth=unbiased_is_truth,
                            whoistruth=whoistruth, who=who, evalset=evalset,
                            clsgroup=clsgroup,
                        )


# %%===========================================================================

if __name__ == '__main__':
    main()
