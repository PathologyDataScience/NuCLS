from scipy.stats import mannwhitneyu, wilcoxon
from pandas import read_csv, concat, read_sql_query
import numpy as np
from os.path import join as opj
from itertools import combinations

from interrater.interrater_utils import _connect_to_anchor_db, \
    get_roc_and_auroc_for_who, remap_classes_in_anchorsdf
from configs.nucleus_style_defaults import Interrater as ir

rpath = '/home/mtageld/Desktop/cTME/results/tcga-nucleus/interrater/CURATED_v1_2020-03-29_EVAL/'  # noqa
rfile = opj(rpath, 'i12_statistical_tests.txt')


def interrater_pvals():
    """Ps-Ps interrater VS Ps-NPs VS NPs-NPs (E-set, classific.)
    Mann-whitney-U"""

    res = ""
    res += "\n**********************************************************************"  # noqa
    res += "\ni9_InterRaterStats -> interrater_boxplots.csv\n"
    res += "\nClassification pairwise inter-rater agreement for the E-set.\n"

    pvals_interrater = {}
    dfs = {}
    for clsgroup in ['main', 'super']:

        # read df
        fpath = opj(rpath, f'i9_InterRaterStats/{clsgroup}/csv/interrater_boxplots.csv')  # noqa
        df = read_csv(fpath, index_col=0)
        df = df.loc[df.loc[:, 'evalset'] == 'E', :]
        dfs[clsgroup] = df

        # comparisons within the same class grouping
        comps = {
            cp: df.loc[df.loc[:, 'comparison'] == cp, 'classification'].values
            for cp in ['Ps-Ps', 'Ps-NPs', 'NPs-NPs']
        }
        for cp1, cp2 in combinations(comps.keys(), 2):
            _, pvals_interrater[f'{cp1} VS {cp2} ({clsgroup})'] = mannwhitneyu(
                comps[cp1], comps[cp2], alternative='two-sided')

    res += f'\n> pvals_interrater (MANNWHITNEYU): '
    res += '----------------------------\n'
    for k, v in pvals_interrater.items():
        res += f"{k.replace('_', ' ')}: %.3f\n" % v

    # comparison across class grouping for same participant pairs
    pvals_interrater = {}
    for cp in ['Ps-Ps', 'Ps-NPs', 'NPs-NPs']:
        compdf = []
        for clsgroup in ['main', 'super']:
            df = dfs[clsgroup]
            df = df.loc[df.loc[:, 'comparison'] == cp, ['classification']]
            df.rename(
                columns={'classification': f'classification_{clsgroup}'},
                inplace=True)
            compdf.append(df)
        compdf = concat(compdf, axis=1, join='inner')
        _, pvals_interrater[f'{cp} (main VS super)'] = wilcoxon(
            compdf.iloc[:, 0], compdf.iloc[:, 1], alternative='two-sided')

    res += f'\n> pvals_interrater (MAIN VS SUPER, WILCOXON): '
    res += '----------------------------\n'
    for k, v in pvals_interrater.items():
        res += f"{k.replace('_', ' ')}: %.3f\n" % v

    print(res)
    with open(rfile, 'a') as f:
        f.write(res)


def intrarater_pvals_accross_evalsets(
        clsgroup: str, method: str, justnps_segm=False, justnps_cls=True):
    """Intra-rater comparison vs the U-control

    wilcoxon: paired, but loss of info because half participants didnt do
    B-control!

    mannwhitneyu: unpaired, but all data can be compared
    """
    assert method in ['wilcoxon', 'mannwhitneyu']
    pvals_intrarater = {}

    # proportion segmented
    fpath = opj(rpath, f'i8_IntraRaterStats/{clsgroup}/csv/intra-rater_comparison.csv')  # noqa
    tmpdf = read_csv(fpath, index_col=0)
    tmpdf = tmpdf.loc[:, ['evalset', 'swho', 'psegmented']]
    if justnps_segm:
        tmpdf = tmpdf.loc[tmpdf.loc[:, 'swho'] == 'NPs', :]  # restrict to NPs?
    bcontrol = tmpdf.loc[tmpdf.loc[:, 'evalset'] == 'B-control', :].dropna()
    bcontrol.columns = [f'bc_{c}' for c in bcontrol.columns]
    eset = tmpdf.loc[tmpdf.loc[:, 'evalset'] == 'E', :].dropna()
    eset.columns = [f'e_{c}' for c in eset.columns]

    if method == 'wilcoxon':
        df = concat([bcontrol, eset], axis=1, join='inner')
        _, pvals_intrarater['psegmented'] = wilcoxon(
            df.loc[:, 'bc_psegmented'],
            df.loc[:, 'e_psegmented'],
            alternative='two-sided')
    elif method == 'mannwhitneyu':
        df = concat([bcontrol, eset], axis=1, join='outer')
        _, pvals_intrarater['psegmented'] = mannwhitneyu(
            df.loc[:, 'bc_psegmented'].dropna(),
            df.loc[:, 'e_psegmented'].dropna(),
            alternative='two-sided')

    # Classification
    fpath = opj(rpath, f'i8_IntraRaterStats/{clsgroup}/csv/intra-rater_comparison.csv')  # noqa
    tmpdf = read_csv(fpath, index_col=0)
    tmpdf = tmpdf.loc[:, ['evalset', 'swho', 'classification']]
    if justnps_cls:
        tmpdf = tmpdf.loc[tmpdf.loc[:, 'swho'] == 'NPs', :]  # restrict to NPs?
    bcontrol = tmpdf.loc[tmpdf.loc[:, 'evalset'] == 'B-control', :].dropna()
    bcontrol.columns = [f'bc_{c}' for c in bcontrol.columns]
    eset = tmpdf.loc[tmpdf.loc[:, 'evalset'] == 'E', :].dropna()
    eset.columns = [f'e_{c}' for c in eset.columns]

    if method == 'wilcoxon':
        df = concat([bcontrol, eset], axis=1, join='inner')
        _, pvals_intrarater['classification'] = wilcoxon(
            df.loc[:, 'bc_classification'],
            df.loc[:, 'e_classification'],
            alternative='two-sided')
    elif method == 'mannwhitneyu':
        df = concat([bcontrol, eset], axis=1, join='outer')
        _, pvals_intrarater['classification'] = mannwhitneyu(
            df.loc[:, 'bc_classification'].dropna(),
            df.loc[:, 'e_classification'].dropna(),
            alternative='two-sided')

    res = ""
    if (clsgroup == 'main') and (method == 'wilcoxon'):
        res += "\n**********************************************************************"  # noqa
        res += "\ni8_IntraRaterStats -> intra-rater_comparison.csv (ACCROSS EVALSETS)\n"  # noqa
        res += "\nIntra-rater comparison vs the U-control.\n"
        res += "- Option 1: Wilcoxon: paired, but loss of info because half participants didnt do B-control.\n"  # noqa
        res += "- Option 2: Mannwhitneyu: unpaired, but all data can be compared.\n"  # noqa
    res += f'\n> pvals_intrarater ({clsgroup.upper()}, {method.upper()}): '
    res += '----------------------------\n'
    for k, v in pvals_intrarater.items():
        res += f'{k}: %.3f\n' % v

    print(res)
    with open(rfile, 'a') as f:
        f.write(res)


def intrarater_pvals_accross_clsgroup(evalset: str):
    """Intra-rater comparison (self agreement vs the U-control) for
    main vs super classes for various pariticpant groups within the same
    evaluation set.
    """
    pvals = {}

    # read dfs
    dfs = {}
    for clsgroup in ['main', 'super']:
        fpath = opj(rpath, f'i8_IntraRaterStats/{clsgroup}/csv/intra-rater_comparison.csv')  # noqa
        df = read_csv(fpath, index_col=0)
        df = df.loc[df.loc[:, 'evalset'] == evalset, :]
        dfs[clsgroup] = df

    for swho in ['Ps', 'NPs']:
        compdf = []
        for clsgroup in ['main', 'super']:
            df = dfs[clsgroup]
            df = df.loc[df.loc[:, 'swho'] == swho, ['classification']]
            df.rename(
                columns={'classification': f'classification_{clsgroup}'},
                inplace=True)
            compdf.append(df)
        compdf = concat(compdf, axis=1, join='inner')
        _, pvals[f'{swho} (main VS super, {evalset}, WILCOXON)'] = wilcoxon(
            compdf.iloc[:, 0], compdf.iloc[:, 1], alternative='two-sided')

    res = ""
    if evalset == 'B-control':
        res += "\n**********************************************************************"  # noqa
        res += "\ni8_IntraRaterStats -> intra-rater_comparison.csv (ACCROSS CLSGROUP)\n"  # noqa
        res += "\nIntra-rater comparison vs the U-control (by class grouping).\n"  # noqa
    res += f'\n> pvals_intrarater ({evalset}): '
    res += '----------------------------\n'
    for k, v in pvals.items():
        res += f'{k}: %.3f\n' % v

    print(res)
    with open(rfile, 'a') as f:
        f.write(res)


def roc_pvals(clsgroup, ntrials=1000, unbiased=False):
    """Accuracy of inferred truth from NPs with/out algorithmic suggestions.

    This gets the bootstrap 95% confidence interval and p-values.
    """
    print(f"\n> [GO GET COFFEE ...] Getting roc_pvals for {clsgroup.upper()})")

    # connect to sqlite database -- anchors
    dbcon = _connect_to_anchor_db(rpath)

    # first we read all anchors
    ubstr = ir._ubstr(unbiased)
    truthcol = f'{ubstr}EM_inferred_label_Ps'
    anchors = {}
    for evalset in ir.MAIN_EVALSET_NAMES:
        # read real anchors and remap labels
        ubstr = ir._ubstr(unbiased)
        tablename = f'v3.1_final_anchors_{evalset}_{ubstr}Ps_AreTruth'
        anchs = read_sql_query(f"""
            SELECT * FROM "{tablename}"
        ;""", dbcon)
        anchs = remap_classes_in_anchorsdf(
            anchors=anchs, clsgroup=clsgroup,
            also_ilabel=True, remove_ambiguous=True,
            who_determines_ambig='Ps', how_ambig_is_determined='EM',
        )

        anchs.loc[:, 'ilabel'] = anchs.loc[:, 'EM_inferred_label_NPs']
        anchors[evalset] = anchs

    # get bootstrap roc aucs
    cats = ['micro', 'macro']
    roc_aucs = {
        cat: {evs: [] for evs in ir.MAIN_EVALSET_NAMES}
        for cat in cats
    }
    for _ in range(ntrials):
        for evalset in ir.MAIN_EVALSET_NAMES:
            x = anchors[evalset]
            idxs = np.random.randint(x.shape[0], size=x.shape[0])
            _, _, rocauc = get_roc_and_auroc_for_who(
                anchors=x.iloc[idxs, :], truthcol=truthcol,
                probcol_prefix='EM_prob_', probcol_postfix='_NPs',
            )
            for cat in cats:
                roc_aucs[cat][evalset].append(rocauc[cat])

    # now get p-values
    pvals = {}
    for ev1, ev2 in combinations(ir.MAIN_EVALSET_NAMES, 2):
        for cat in cats:
            _, pvals[f'{ev1}_VS_{ev2}_{cat}'] = mannwhitneyu(
                roc_aucs[cat][ev1], roc_aucs[cat][ev2],
                alternative='two-sided')

    res = ""
    if clsgroup == 'main':
        res += "\n**********************************************************************"  # noqa
        res += "\ni5_ParticipantAccuracy -> Ps_AreTruth_superimposed_auroc_curves.svg\n"  # noqa
        res += "\nAccuracy of inferred truth from NPs with/out algorithmic suggestions.\n"  # noqa
        res += f"This is the bootstrap AUROC comparison p-value with {ntrials} trials.\n"  # noqa
    res += f'\n> AUROCs ({clsgroup.upper()}): '
    res += '----------------------------\n'
    for cat, aucvals_dict in roc_aucs.items():
        for ev, aucvals in aucvals_dict.items():
            res += (
                f"{cat}: {ev}: {np.round(np.percentile(aucvals, 50), 3)} "
                f"({np.round(np.percentile(aucvals, 5), 3)}, "
                f"{np.round(np.percentile(aucvals, 95), 3)})\n"
            )

    res += f'\n> pvals_intrarater ({clsgroup.upper()}, MANNWHITNEYU): '
    res += '----------------------------\n'
    for k, v in pvals.items():
        res += f"{k.replace('_', ' ')}: %.3f\n" % v

    print(res)
    with open(rfile, 'a') as f:
        f.write(res)


def segmentation_pvals(who='NPs', metric='DICE'):
    """Segmentation accuracy p-values"""
    fpath = opj(rpath, f'i6_SegmentationAccuracy/{who}_AreTruth/csv/'
                       f'{who}_AreTruth_evalset_violinplot_comparison.csv')
    df = read_csv(fpath)

    comps = {}
    eset = df.loc[df.loc[:, 'evalset'] == 'E', :]
    bset = df.loc[df.loc[:, 'evalset'] == 'B-control', :]

    _, comps['E-set VS B-control (overall)'] = mannwhitneyu(
        eset.loc[:, metric].values,
        bset.loc[:, metric].values,
        alternative='two-sided')

    _, comps['E-set VS B-control (correct)'] = mannwhitneyu(
        eset.loc[eset.loc[:, 'iscorrect'] == 1, metric].values,
        bset.loc[bset.loc[:, 'iscorrect'] == 1, metric].values,
        alternative='two-sided')

    _, comps['E-set VS B-control (incorrect)'] = mannwhitneyu(
        eset.loc[eset.loc[:, 'iscorrect'] == 0, metric].values,
        bset.loc[bset.loc[:, 'iscorrect'] == 0, metric].values,
        alternative='two-sided')

    _, comps['E-set correct vs incorrect'] = mannwhitneyu(
        eset.loc[eset.loc[:, 'iscorrect'] == 1, metric].values,
        eset.loc[eset.loc[:, 'iscorrect'] == 0, metric].values,
        alternative='two-sided')

    _, comps['B-control correct vs incorrect'] = mannwhitneyu(
        bset.loc[bset.loc[:, 'iscorrect'] == 1, metric].values,
        bset.loc[bset.loc[:, 'iscorrect'] == 0, metric].values,
        alternative='two-sided')

    res = ""
    res += "\n**********************************************************************"  # noqa
    res += f"\ni6_SegmentationAccuracy -> {who}_AreTruth_evalset_violinplot_comparison.csv\n"  # noqa
    res += "\nSegmentation accuracy p-values.\n"
    res += f'\n> segmentation_pvals ({metric}), MANNWHITNEYU): '
    res += '----------------------------\n'
    for k, v in comps.items():
        res += f"{k}: %.3f\n" % v

    # same nucleus in both evalsets
    fpath = opj(rpath, f'i6_SegmentationAccuracy/{who}_AreTruth/csv/'
                       f'{who}_AreTruth_evalset_{metric}_comparison.csv')
    df = read_csv(fpath, index_col=0)
    _, pval = wilcoxon(
        df.loc[:, f'{metric}_B-control'], df.loc[:, f'{metric}_E'],
        alternative='two-sided')
    res += f'\n> segmentation_pvals, joint ({metric}), WILCOXON): '
    res += '----------------------------\n'
    res += f"E-set VS B-control (joint): %.3f\n" % pval

    # save
    print(res)
    with open(rfile, 'a') as f:
        f.write(res)


# =============================================================================

if __name__ == '__main__':

    def run_seq1(fun, **kwargs):
        for clsgroup in ['main', 'super']:
            fun(clsgroup=clsgroup, **kwargs)

    def run_seq2(fun, **kwargs):
        for evalset in ['B-control', 'E']:
            fun(evalset=evalset, **kwargs)

    # interrater_pvals()
    # run_seq1(intrarater_pvals_accross_evalsets, method='wilcoxon')
    # run_seq1(intrarater_pvals_accross_evalsets, method='mannwhitneyu')
    # run_seq2(intrarater_pvals_accross_clsgroup)
    # segmentation_pvals()

    # This takes a while
    run_seq1(roc_pvals)
