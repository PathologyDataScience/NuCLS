import sys
import os
from os.path import join as opj
import matplotlib.pylab as plt
import numpy as np
from pandas import read_csv


def plot_losses(bygrup, byepoch, axis, plot_grup=True):
    loss_styles = {
        'loss': {'color': '#000066', 'ls': '-'},
        'loss_objectness': {'color': '#006600', 'ls': '-'},
        'loss_box_reg': {'color': '#663300', 'ls': '-'},
        'loss_rpn_box_reg': {'color': '#996633', 'ls': '-'},
        'loss_mask': {'color': '#990000', 'ls': '-'},
        'loss_classifier': {'color': '#ff9900', 'ls': '-'},
    }
    for loss_type, loss_style in loss_styles.items():
        if loss_type in byepoch.columns:
            # by batch
            if plot_grup:
                axis.plot(
                    bygrup.loc[:, 'gradient_update'],
                    bygrup.loc[:, loss_type],
                    alpha=0.2, **loss_style)
            # by epoch
            axis.plot(
                byepoch.loc[:, 'gradient_update'],
                byepoch.loc[:, loss_type],
                # marker='.',
                label=loss_type, **loss_style)

    axis.set_xlabel('gradient_update')
    axis.set_ylabel('loss')
    axis.legend()


def plot_mAP(tsmetrics, axis, mprefix=''):
    metric_styles = {
        'mAP @ 0.50:0.95': {'color': '#006699', 'linestyle': '-'},
        'AP @ 0.5': {'color': '#000066', 'linestyle': '-'},
        'AP @ 0.75': {'color': '#000066', 'linestyle': '-', 'alpha': 0.2},
        'segm mAP @ 0.50:0.95': {'color': '#ff5050', 'ls': '-'},
        'segm AP @ 0.5': {'color': '#990000', 'ls': '-'},
        'segm AP @ 0.75': {'color': '#ff8080', 'ls': '-'},
    }
    metric_styles = {mprefix + k: v for k, v in metric_styles.items()}
    for metric, mstyle in metric_styles.items():
        if metric in tsmetrics.columns:
            axis.plot(
                tsmetrics.loc[:, 'epoch'],
                tsmetrics.loc[:, metric],
                marker='.', label=metric, **mstyle)

    axis.set_xlabel('epoch')
    axis.set_ylabel('AP')
    axis.legend()


def plot_accuracy(tsmetrics, axis, prefix=''):
    sctg = 'superCateg_'
    metric_styles = {
        f'{prefix}accuracy': {'color': 'k', 'linestyle': '-'},
        f'{prefix}auroc_micro': {'color': '#000066', 'linestyle': '-'},
        f'{prefix}auroc_macro': {'color': '#006699', 'linestyle': '-'},
        f'{prefix}mcc': {'color': '#990000', 'linestyle': '-'},
    }
    if prefix == sctg:
        # breakdown by supercategory
        metric_styles.update({
            f'{sctg}aucroc_tumor_any': {'color': [1., 0., 0.], 'ls': '-', 'alpha': 0.2},
            f'{sctg}aucroc_nonTIL_stromal': {'color': [0., 0.9, 0.3], 'ls': '-', 'alpha': 0.2},
            f'{sctg}aucroc_sTIL': {'color': [0., 0., 1.], 'ls': '-', 'alpha': 0.2},
            f'{sctg}other_nucleus': {'color': 'gray', 'ls': '-', 'alpha': 0.2},
        })
    for metric, mstyle in metric_styles.items():
        if metric in tsmetrics.columns:
            axis.plot(
                tsmetrics.loc[:, 'epoch'],
                tsmetrics.loc[:, metric],
                marker='.', label=metric, **mstyle)

    axis.set_xlabel('epoch')
    axis.set_ylabel('accuracy')
    axis.legend()


def plot_accuracy_progress(checkpoint_path, postfix='', plotdet=True):

    # read metrics
    base = checkpoint_path.split('.ckpt')[0]
    bygrup = read_csv(base + '_trainingLossByGrUp.csv')
    byepoch = read_csv(base + '_trainingLossByEpoch.csv')
    tsmetrics = read_csv(base + f'_testingMetricsByEpoch{postfix}.csv')

    classification = 'accuracy' in tsmetrics.columns
    nperrow = 5 if classification else 2
    nrows = 1
    fig, ax = plt.subplots(nrows, nperrow, figsize=(5 * nperrow, 5.3 * nrows))

    # training losses
    axis = ax[0]
    plot_losses(bygrup=bygrup, byepoch=byepoch, axis=axis, plot_grup=False)
    axis.set_title("Training loss", fontsize=12, fontweight='bold')

    # testing metrics -- AP (combined detection & classification)
    if plotdet:
        axis = ax[1]
        plot_mAP(tsmetrics=tsmetrics, axis=axis)
        axis.set_title(
            "Testing accuracy (det. & class.)", fontsize=12, fontweight='bold')

    if classification:

        # testing metrics -- AP (only detection)
        if plotdet:
            axis = ax[2]
            plot_mAP(tsmetrics=tsmetrics, axis=axis, mprefix='objectness ')
            axis.set_title(
                "Testing accuracy (objectness)", fontsize=12,
                fontweight='bold')

        # testing metrics -- classification
        axis = ax[3]
        plot_accuracy(tsmetrics=tsmetrics, axis=axis)
        axis.set_title(
            "Testing accuracy (classif., as-is)",
            fontsize=12, fontweight='bold')

        # testing metrics -- classification (supercategories)
        axis = ax[4]
        plot_accuracy(tsmetrics=tsmetrics, axis=axis, prefix='superCateg_')
        axis.set_title(
            "Testing accuracy (classif., superCategs)",
            fontsize=12, fontweight='bold')

    plt.savefig(base + f'_trainingAcc{postfix}.svg')
    plt.close()


def vis_bbox(img, bbox, label=None, score=None, label_names=None,
             instance_colors=None, alpha=1., linewidth=1., ax=None):
    """Visualize bounding boxes inside the image.

    Modified from: https://github.com/DeNA/PyTorch_YOLOv3/blob/master/utils/vis_bbox.py
    Args:
        img (~numpy.ndarray): An array of shape :math:`(height, width, 3)`.
            This is in RGB format and the range of its value is
            :math:`[0, 255]`. If this is :obj:`None`, no image is displayed.
        bbox (~numpy.ndarray): An array of shape :math:`(R, 4)`, where
            :math:`R` is the number of bounding boxes in the image.
            Each element is organized
            by :math:`(x_{min}, y_{min}, x_{max}, y_{max})` in the second axis.
        label (~numpy.ndarray): An integer array of shape :math:`(R,)`.
            The values correspond to id for label names stored in
            :obj:`label_names`. This is optional.
        score (~numpy.ndarray): A float array of shape :math:`(R,)`.
             Each value indicates how confident the prediction is.
             This is optional.
        label_names (iterable of strings): Name of labels ordered according
            to label ids. If this is :obj:`None`, labels will be skipped.
        instance_colors (iterable of tuples): List of colors.
            Each color is RGB format and the range of its values is
            :math:`[0, 255]`. The :obj:`i`-th element is the color used
            to visualize the :obj:`i`-th instance.
            If :obj:`instance_colors` is :obj:`None`, the red is used for
            all boxes.
        alpha (float): The value which determines transparency of the
            bounding boxes. The range of this value is :math:`[0, 1]`.
        linewidth (float): The thickness of the edges of the bounding boxes.
        ax (matplotlib.axes.Axis): The visualization is displayed on this
            axis. If this is :obj:`None` (default), a new axis is created.
    Returns:
        ~matploblib.axes.Axes:
        Returns the Axes object with the plot for further tweaking.
    from: https://github.com/chainer/chainercv
    """

    if label is not None and not len(bbox) == len(label):
        raise ValueError('The length of label must be same as that of bbox')
    if score is not None and not len(bbox) == len(score):
        raise ValueError('The length of score must be same as that of bbox')

    # Returns newly instantiated matplotlib.axes.Axes object if ax is None
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
    ax.imshow(np.uint8(img))
    # If there is no bounding box to display, visualize the image and exit.
    if len(bbox) == 0:
        return ax

    if instance_colors is None:
        # yellow
        instance_colors = np.zeros((len(bbox), 3), dtype=np.float32)
        instance_colors[:, 0] = 255
        instance_colors[:, 1] = 255
    instance_colors = np.array(instance_colors)

    for i, bb in enumerate(bbox):
        xy = (bb[0], bb[1])
        height = bb[3] - bb[1]
        width = bb[2] - bb[0]
        color = instance_colors[i % len(instance_colors)] / 255
        ax.add_patch(plt.Rectangle(
            xy, width, height, fill=False,
            edgecolor=color, linewidth=linewidth, alpha=alpha))

        caption = []

        if label is not None and label_names is not None:
            lb = label[i]
            if not (0 <= lb < len(label_names)):
                raise ValueError('No corresponding name is given')
            caption.append(label_names[lb])
        if score is not None:
            sc = score[i]
            caption.append('{:.2f}'.format(sc))

        if len(caption) > 0:
            ax.text(bb[1], bb[0],
                    ': '.join(caption),
                    style='italic',
                    bbox={'facecolor': 'white', 'alpha': 0.7, 'pad': 10})
    return ax


def scatter_embeddings(
        input_df,
        embedding_columns=None,
        classes_column="true_categ",
        classes_set=None,
        color_dict=None,
        color_by_column=None,
        pltcmap='hot',
        title=None, dest_path="./outputs", savename="scatter",
        alphas=None, init_point_size=25.0, point_size_ds=0.9,
        axes_ticks=True, xmin=None, xmax=None, ymin=None, ymax=None,):
    """
    Modified from: Pooya Mobadersay.

    This function generates the scatterplots for embeddings.

    Parameters
    ----------
    input_df: Pandas DataFrame
      e.g. loaded nucleus_metadata_and_embeddings.csv file

    embedding_columns: list of strings
      list of column names of embeddings in all_results

    classes_column: string
      name of the column that contains class labels

    classes_set: list of strings
      List of desired classes to visualize (in the same order to vis.)

    color_dict: dictionary
      dictionary that maps class names to their desired colors in visualiz.

    color_by_column: str
      name of column to color by (color map intensity based on this column)

    axis_ticks: Boolean
      Whether you want to include the axes ticks in the figure or not

    title: String
      This will be the title of the figure

    dest_path: String
      The destination path to save the results

    alphas: list of floats with only two elements [a, b]
      a, b should be in [0.0, 1.0] interval
      This defines the desired alpha channels' interval across all the classes.
      The first value will be assigned to the first class and the last value
      will be assigned to the last class; any class in between will get alphas
      with constant difference inside the interval.

      if a>b then the alphas will decrease from the first class to the end
      if a<b then the alphas will increase from the first class to the end
      if a==b then the alphas will be constant from the first class to the end

      Lower alpha means more transparency.

    init_point_size: Float
      The initial data point size assigned to the first class

    point_size_ds: Float
      Point size Down Scaling factor, the point sizes will decrease with this
      scale from one class to other. If you want to have same data point sizes
      across all classes make this value 1.0


    Example
    ----------
    color_dict = {'tumor_nonMitotic': [255, 0, 0],
                  'tumor_mitotic': [255, 191, 0],
                  'nonTILnonMQ_stromal': [0, 230, 77],
                  'macrophage': [51, 102, 153],
                  'lymphocyte': [0, 0, 255],
                  'plasma_cell': [0, 255, 255],
                  'other_nucleus': [0, 0, 0],
                  'AMBIGUOUS': [80, 80, 80]
    draw_scatter(all_results="./all_results.csv", color_dict=color_dict)
    """
    assert color_dict or color_by_column
    embedding_columns = embedding_columns or ['embedding_0', 'embedding_1']
    classes_set = classes_set or [
        'macrophage',
        'tumor_nonMitotic',
        'lymphocyte',
        'tumor_mitotic',
        'nonTILnonMQ_stromal',
        'plasma_cell'
    ]
    alphas = alphas or [0.8, 0.4]

    plt.figure(figsize=(7,7))

    embedding = input_df.loc[:, embedding_columns].values
    classes = input_df.loc[:, classes_column].values

    # normalize the specific feature to be used for coloring/alpha
    if color_by_column is not None:
        colby = input_df.loc[:, color_by_column].values
        colby = (colby - np.nanmean(colby)) / np.nanstd(colby)
        colby -= np.nanmin(colby)
        colby /= np.nanmax(colby)
    else:
        colby = np.ones((input_df.shape[0],), dtype='float')

    # determine colors
    if color_dict is not None:
        # different colors for different nucleus classes
        color_map = np.float32(input_df.loc[:, classes_column].map(color_dict).tolist())
        color_map /= 255.
        color_map = np.concatenate((color_map, colby[:, None]), axis=1)
    else:
        # the color itself is determined by a specific feature
        from matplotlib import cm
        cmp = cm.get_cmap(pltcmap, 100)
        color_map = cmp(colby)

    # defined to make the points more transparent for overlayed scatterplots
    alphas = np.linspace(alphas[0], alphas[1], len(classes_set))
    point_size = init_point_size
    if not os.path.exists(dest_path):
        os.makedirs(dest_path)
    x = embedding[:, 0]
    y = embedding[:, 1]
    for (i, cla) in enumerate(classes_set):
        keep = classes == cla
        colors = color_map[keep]
        colors[:, 3] = colors[:, 3] * alphas[i]
        plt.scatter(
            x[keep], y[keep], c=color_map[keep], label=cla,
            s=point_size, edgecolors='none')
        point_size = point_size_ds * point_size

    if not axes_ticks:
        plt.xticks([], [])
        plt.yticks([], [])
    if all([j is not None for j in (xmin, xmax, ymin, ymax)]):
        plt.xlim(xmin, xmax)
        plt.ylim(ymin, ymax)
    # plt.xlabel(embedding_columns[0])
    # plt.ylabel(embedding_columns[1])
    plt.title(label=title, fontsize=14, fontweight='bold')
    # if color_dict is not None:
    #     plt.legend()

    plt.savefig(opj(dest_path, f'{savename}_{color_by_column}.png'))
    # plt.savefig(opj(dest_path, f'{savename}_{color_by_column}.svg'))
    plt.close()
