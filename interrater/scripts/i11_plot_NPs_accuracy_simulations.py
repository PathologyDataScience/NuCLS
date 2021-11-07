from os.path import join as opj
import numpy as np
import matplotlib.pylab as plt
import seaborn as sns
from pandas import read_sql_query

from configs.nucleus_style_defaults import Interrater as ir
from configs.nucleus_model_configs import VisConfigs
from interrater.interrater_utils import _maybe_mkdir, _connect_to_anchor_db, _get_clmap


def plot_simulation_stats(
        dbcon, savedir: str, evalset: str, clsgroup: str):
    """"""
    _maybe_mkdir(opj(savedir, 'csv'))
    _maybe_mkdir(opj(savedir, 'plots'))

    # control for total no of unique pathologists
    nnps = 18 if evalset == 'E' else 19

    # get simulation stats for evalset
    stats = read_sql_query(f"""
        SELECT *
        FROM "NPs_AccuracySimulations_{clsgroup}ClassGroup"
        WHERE "evalset" = "{evalset}"
          AND "n_unique_NPs" = {nnps}
    ;""", dbcon)

    # organize canvas and plot
    _, tmp_classes = _get_clmap(clsgroup)
    tmp_classes.remove('AMBIGUOUS')
    classes = ['detection', 'classification', 'micro', 'macro'] + tmp_classes
    nperrow = 4
    nrows = int(np.ceil((len(classes)) / nperrow))
    fig, ax = plt.subplots(nrows, nperrow, figsize=(5 * nperrow, 5.5 * nrows))
    axno = -1
    for axis in ax.ravel():
        axno += 1

        cls = classes[axno]
        isdetection = cls == 'detection'
        default_color = ir.PARTICIPANT_STYLES['NPs']['c']

        if isdetection:
            metric = 'detection-F1'
            lab = 'Detection F1 score'
            # ymin = 0.
            ymin = 0.5
            color = default_color
        elif cls == 'classification':
            metric = 'classification-all-MCC'
            lab = 'Classification MCC'
            ymin = 0
            color = default_color
        else:
            if cls == 'micro':
                clstr = 'Micro-Average'
                color = default_color
            elif cls == 'macro':
                clstr = 'Macro-Average'
                color = default_color
            else:
                clstr = cls.capitalize()
                color = [j / 255. for j in VisConfigs.CATEG_COLORS[cls]]
            metric = f'auroc-{cls}'
            lab = f'AUROC - {clstr}'
            ymin = 0.5

        # main boxplot
        bppr = {'alpha': 0.6, 'color': color}
        sns.boxplot(
            ax=axis, data=stats, x='NPs_per_fov', y=metric,
            boxprops=bppr, whiskerprops=bppr, capprops=bppr, medianprops=bppr,
            showfliers=False, color=color,
            notch=True, bootstrap=5000,
            # notch=False,
        )

        axis.set_ylim(ymin, 1.)
        axis.set_title(lab, fontsize=14, fontweight='bold')
        axis.set_ylabel(lab, fontsize=11)
        axis.set_xlabel('No. of NPs per FOV', fontsize=11)

    plt.tight_layout(pad=0.3, w_pad=0.5, h_pad=0.3)
    savename = f'NPs_AccuracySimulation_PsAreTruth_{evalset}'
    plt.savefig(opj(savedir, 'plots', savename + '.svg'))
    plt.close()


def main():

    DATASETNAME = 'CURATED_v1_2020-03-29_EVAL'

    # where to save stuff
    BASEPATH = "/home/mtageld/Desktop/cTME/results/tcga-nucleus/interrater/"
    SAVEDIR = opj(BASEPATH, DATASETNAME, 'i11_NPsAccuracySimulations')
    _maybe_mkdir(SAVEDIR)

    # connect to sqlite database -- anchors
    dbcon = _connect_to_anchor_db(opj(SAVEDIR, '..'))

    # Go through various evaluation sets & participant groups
    for evalset in ['E']:
        for clsgroup in ['super']:
            savedir = opj(SAVEDIR, clsgroup)
            _maybe_mkdir(savedir)
            plot_simulation_stats(
                dbcon=dbcon, savedir=savedir, evalset=evalset,
                clsgroup=clsgroup)


# %%===========================================================================

if __name__ == '__main__':
    main()
