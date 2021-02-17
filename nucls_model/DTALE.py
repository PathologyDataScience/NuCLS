import sys
from os.path import join as opj
from collections import Counter
import numpy as np
import pickle
import matplotlib.pylab as plt
from sklearn.metrics import matthews_corrcoef
from sklearn.tree import DecisionTreeRegressor

from configs.nucleus_model_configs import VisConfigs
from GeneralUtils import maybe_mkdir


def calc_stats_simple(TP, FP, FN, TN=None, add_eps_to_tn=True):
    """Calculate simple stistics"""
    ep = 1e-10
    if TP == 0:
        TP += ep
    if FP == 0:
        FP += ep
    if FN == 0:
        FN += ep
    TN = 0 if TN is None else TN

    stats = {'total': TP + FP + FN + TN}
    stats.update({
        'TP': TP,
        'FP': FP,
        'FN': FN,
        'accuracy': (TP + TN) / stats['total'],
        'precision': TP / (TP + FP),
        'recall': TP / (TP + FN),
    })
    # add synonyms
    stats.update({
        'TPR': stats['recall'],
        'sensitivity': stats['recall'],
        'F1': (2 * stats['precision'] * stats['recall']) / (
                stats['precision'] + stats['recall']),
    })
    if TN >= 0:
        if TN == 0:
            if add_eps_to_tn:
                TN += ep
            else:
                return stats
        stats.update({
            'TN': TN,
            'specificity': TN / (TN + FP)
        })
        # add synonyms
        stats['TNR'] = stats['specificity']

        # mathiew's correlation coefficient
        numer = TP * TN - FP * FN
        denom = np.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
        stats['MCC'] = numer / denom

    return stats


class DTALE(object):
    """
    A decision tree the learns to make same predictions as the nucleus model
    classifier, but using interpretable features. Note that the tree is
    learning using the mask-rcnn model predictions, NOT the gtruth. The idea is
    to find an interpretable approximation of what the classification component
    of the model seems to be relying on. We rely on a REGRESSION tree for a
    more refined approximation of the model behavior.

    I decided to call this technique DTALE, which stands for

     > Decision Tree Approximation of Learned Embeddings (DTALE)

    Make sure to take a look and cite these references in the paper for
    some context, but I haven't seen this exact way of doing things in the
    deep-learning literature, and definitely not in the computational pathology
    space:

    - Dahlin N, Kalagarla KC, Naik N, Jain R, Nuzzo P. Designing Interpretable
      Approximations to Deep Reinforcement Learning with Soft Decision Trees.
      arXiv preprint arXiv:2010.14785. 2020 Oct 28.
    - https://lilianweng.github.io/lil-log/2017/08/01/how-to-explain-the- ...
      prediction-of-a-machine-learning-model.html

    """
    def __init__(
            self, feats, clusts, savedir,
            pcoln='pred_categ', ecoln0='embedding_0', ecoln1='embedding_1',
            classes_list=None, fitkwargs=None):

        # drop nans
        clusts = clusts.dropna(axis=0)
        feats = feats.loc[clusts.index, :]
        feats = feats.dropna(axis=0)
        clusts = clusts.loc[feats.index, :]
        self.feats = feats
        self.clusts = clusts

        # some ground work
        self.pcoln = pcoln
        self.ecoln0 = ecoln0
        self.ecoln1 = ecoln1
        _, y = self._getxy()
        self._e0min = y[:, 0].min()
        self._e0max = y[:, 0].max()
        self._e1min = y[:, 1].min()
        self._e1max = y[:, 1].max()

        # assign params or defaults
        self.classes_list = classes_list or list(
            set(clusts.loc[:, pcoln].tolist()))
        self.fitkwargs = fitkwargs or {
            'random_state': 0,
            'min_samples_leaf': 250,  # best: 250
            'max_depth': 7,  # best: 7
        }

        # init attribs
        self.savedir = savedir
        self.featnames = np.array(feats.columns)
        self.model = None
        self.tree = None
        self.n_nodes = None
        self.pred_y_leafs = None
        self.leafs = None
        self.nodes = None
        self.node_leafs = {}
        self.node_tally = {}

    def _getxy(self):
        X = self.feats.values
        y = self.clusts.loc[:, [self.ecoln0, self.ecoln1]].values
        return X, y

    def fit_model(self):
        """Fit a DTALE model."""

        # fit regressor to predict embeddings from NuCLS model
        self.model = DecisionTreeRegressor(**self.fitkwargs)
        X, y = self._getxy()
        self.model.fit(X, y)
        self.tree = self.model.tree_

        # save model for reproducibility
        with open(opj(self.savedir, 'dectree.pkl'), 'wb') as f:
            pickle.dump(self.model, f)

        # # load model
        # with open(opj(savedir, 'dectree.pkl'), 'rb') as f:
        #     loaded_model = pickle.load(f)

        # # show tree text
        # r = export_text(regr, feature_names=list(feats.columns))
        # print(r)

    def apply_model(self):

        self.n_nodes = self.tree.node_count
        self.leafs = np.argwhere(self.tree.children_left == -1)[:, 0].tolist()
        self.nodes = {i for i in range(self.n_nodes)}.difference(self.leafs)

        # Apply to training data
        X, _ = self._getxy()
        self.pred_y_leafs = self.model.apply(X)
        # self.pred_y_vals = self.tree.value[self.pred_y_leafs, :, 0]

    def _find_leaves_in_subtree(self, root, subtrees):
        """find all the leaves enclosed within a subtree."""

        leafs = []

        def _traverse(node):

            # dynamic programming
            if node in subtrees:
                leafs.extend(subtrees[node])
                return
            subtrees[node] = []

            children = (
                self.tree.children_left[node],
                self.tree.children_right[node],
            )
            if children[0] == -1:
                leafs.append(node)
                subtrees[node].append(node)
                return
            for child in children:
                _traverse(child)

        _traverse(node=root)
        subtrees[root] = leafs

        return subtrees

    def set_leafs_for_all_subtrees(self):
        """Get all subleafs enclosed within each node subtree."""
        # traverse from bottom up for dynamic programming speedup
        for nd in range(self.n_nodes - 1, -1, -1):
            self._find_leaves_in_subtree(root=nd, subtrees=self.node_leafs)

    def set_node_tally(self):
        """
        Get a tally of the number of points from each class (as determined
        by the NuCLS model final prediction) for each node.
        """
        self.node_tally = {
            leaf: Counter(self.clusts.loc[
                self.pred_y_leafs == leaf, self.pcoln].to_list())
            for leaf in self.leafs
        }
        for node, nlfs in self.node_leafs.items():
            if node in self.leafs:
                continue
            self.node_tally[node] = self.node_tally[nlfs[0]]
            for nlf in nlfs[1:]:
                self.node_tally[node] += self.node_tally[nlf]

    def _get_best_node_for_class(self, cls, metric):
        """
        For one class, find the cluster (node) which overlaps the most
        with the predictions from the NuCLS model
        """
        best_node = None
        best_stats = {metric: -1. if metric == 'MCC' else 0.}
        for node in self.nodes:
            innode = 0 + np.in1d(self.pred_y_leafs, self.node_leafs[node])
            incls = 0 + (self.clusts.loc[:, self.pcoln] == cls).values
            stats = calc_stats_simple(
                TP=np.sum(innode + incls == 2),
                FP=np.sum(innode - incls == 1),
                TN=np.sum(innode + incls == 0),
                FN=np.sum(innode - incls == -1),
            )
            stats['MCC'] = matthews_corrcoef(y_true=incls, y_pred=innode)
            if stats[metric] > best_stats[metric]:
                best_node = node
                best_stats.update(stats)
        return best_node, best_stats

    def _get_best_node_for_each_class(self, metric='precision'):
        """
        For each class, find the cluster (node) which best fits/explains
        predictions from the NuCLS model

        IMPORTANT NOTE:
          The classes are INDEPENDENT of each other. So an early "tumor" node
          does NOT exclude the descendent "mitotic" node. This is EXPECTED
          and cannot be overcome because the nodes are not pure .. even the
          downstream "mitotic" node contains some tumor leafs, so excluding
          it would reduce recall of the "tumor" node. Best way is to think of
          these paths as being independent for different classes.
        """
        best_nodes = {}
        best_stats = {}
        for cls in self.classes_list:
            best_nodes[cls], best_stats[cls] = self._get_best_node_for_class(
                cls, metric=metric)
        return best_nodes, best_stats

    def _trace_from_node_to_root(self, node):
        trace = [node]
        direction = [0]
        current_node = node
        keep_going = True
        while keep_going:
            left = np.argwhere(self.tree.children_left == current_node)
            right = np.argwhere(self.tree.children_right == current_node)
            if len(left) > 0:
                current_node = left[0, 0]
                trace.append(current_node)
                direction.append(-1)
            elif len(right) > 0:
                current_node = right[0, 0]
                trace.append(current_node)
                direction.append(1)
            else:
                keep_going = False
        return trace, direction

    def save_dectree_traces(self, best_nodes, best_stats, postfix=''):
        """Save decision tree traces for relevant classes."""

        node_trace = {}
        direction_trace = {}
        feat_trace = {}
        impurity_trace = {}
        nsize_trace = {}
        thresh_trace = {}
        nice_trace = {}

        for cls in self.classes_list:

            # track from node to root
            ntrace, dtrace = self._trace_from_node_to_root(best_nodes[cls])
            ntrace, dtrace = ntrace[::-1], dtrace[::-1]
            node_trace[cls], direction_trace[cls] = ntrace, dtrace

            # map nodes to feature names and thresholds
            feat_trace[cls] = self.featnames[
                self.tree.feature[ntrace]].tolist()
            impurity_trace[cls] = self.tree.impurity[ntrace].tolist()
            nsize_trace[cls] = self.tree.n_node_samples[ntrace].tolist()
            thresh_trace[cls] = self.tree.threshold[ntrace].tolist()

            # render into nice text
            descr = '\nDECISIONS:\n'
            descr += "--------------\n"
            for nix in range(len(ntrace) - 1):
                dhere = ' '.join([
                    feat_trace[cls][nix], '<=' if dtrace[nix] == -1 else '>',
                    '%.1f' % thresh_trace[cls][nix]])
                descr += dhere + '\n'
            descr += f'\nSTATS:\n'
            descr += "--------------\n"
            descr += '\n'.join([
                f'{st}: %.2f' % stv
                for st, stv in best_stats[cls].items()
            ]) + '\n'
            nice_trace[cls] = descr

        # parse into a dict and pickle
        with open(opj(
                self.savedir, f'dectree_traces{postfix}.pkl'), 'wb') as f:
            pickle.dump({
                'features': feat_trace,
                'thresholds': thresh_trace,
                'impurity': impurity_trace,
                'nodes': node_trace,
                'direction': direction_trace,
                'node_n_samples': nsize_trace,
                'nice': nice_trace,

                # How well the "chosen" traces from our decision tree
                # fit/explain the NuCLS model predictions. For example, a
                # precision of 0.9 for 'tumor' means that 90% of the nuclei
                # predicted as 'tumor' by our decision tree are also predicted
                # as 'tumor' by the NuCLS model.
                'fit_stats_to_NuCLS_model': best_stats,
            }, f)

        # save nice rendered text for relevant parts of tree
        with open(opj(self.savedir, f'dectree_nice{postfix}.txt'), 'w') as f:
            for cls in self.classes_list:
                f.write(
                    "***********************************\n"
                    f"{cls}\n"
                    "***********************************\n"
                )
                f.write(nice_trace[cls] + '\n')

    def visualize_decision_tree_nodes(self, best_nodes, postfix=''):
        """Visualize the learned decision tree nodes."""

        plt.figure(figsize=(7, 7))

        # scatter actual points from NuCLS model in background
        _, y = self._getxy()
        plt.scatter(
            y[:, 0], y[:, 1],
            c='beige', alpha=0.6, s=4, edgecolors='none')

        # trace the learned decision tree
        for node in range(self.tree.node_count):
            if self.tree.children_left[node] == -1:
                continue
            me = self.tree.value[node, :, 0]
            clt = self.tree.value[self.tree.children_left[node], :, 0]
            crt = self.tree.value[self.tree.children_right[node], :, 0]
            plt.plot(
                [clt[0], me[0], crt[0]], [clt[1], me[1], crt[1]],
                color='gray', marker='.', linestyle='-',
                linewidth=0.5, markersize=3, alpha=0.5,
            )

        # highligh root node
        me = self.tree.value[0, :, 0]
        plt.scatter(
            [me[0]], [me[1]],
            color='k', s=30, alpha=1., edgecolors='k')

        # color best (class-representative) nodes by class
        for cls, node in best_nodes.items():

            me = self.tree.value[node, :, 0]

            # color the trace along the decision tree till best node
            trace, _ = self._trace_from_node_to_root(node)
            for ndi in range(len(trace) - 1):
                clt = self.tree.value[trace[ndi], :, 0]
                crt = self.tree.value[trace[ndi + 1], :, 0]
                plt.plot(
                    [clt[0], crt[0]], [clt[1], crt[1]],
                    color='k', alpha=1.,
                    marker='o', markersize=2.5,
                    linestyle='-', linewidth=1.3,
                )

            # highlight actual chosen best node
            color = np.array(VisConfigs.CATEG_COLORS[cls])[None, :] / 255.
            plt.scatter(
                [me[0]], [me[1]],
                color=color, s=150, alpha=1., edgecolors='none')

        plt.xlim(self._e0min, self._e0max)
        plt.ylim(self._e1min, self._e1max)
        plt.title(
            f'DTALE nodes ({postfix})', fontsize=14, fontweight='bold')
        # plt.show()
        # plt.savefig(opj(self.savedir, f'dectree{postfix}.svg'))
        plt.savefig(opj(self.savedir, f'dectree{postfix}.png'))

    def visualize_decision_tree_classes(
            self, best_nodes, classes_list=None,
            restrict_to_pcateg=False, exclude_leafs=None,
            savedir=None, postfix=''):
        """Visualize embeddings, colors by class predicted by decision tree."""

        classes_list = classes_list or self.classes_list
        savedir = savedir or self.savedir

        init_point_size = 10.
        point_size_ds = 1.
        alphas = [0.8, 0.5]
        _, y = self._getxy()

        plt.figure(figsize=(7, 7))

        point_size = init_point_size
        alphas = np.linspace(alphas[0], alphas[1], len(classes_list))

        # keep track of plotted indices to be able to exclude downstream
        # nodes when plotting upstream ones when relevant
        kept_idxs = []

        for clno, cls in enumerate(classes_list):

            # maybe restrict to leafs predicted as a particular class by NuCLS
            keep1 = None
            if restrict_to_pcateg:
                keep1 = (self.clusts.loc[:, 'pred_categ'] == cls).values

            # restrict to downstream leafs to node of interest
            keep2 = np.in1d(self.pred_y_leafs, self.node_leafs[best_nodes[cls]])  # noqa
            if keep1 is None:
                keep = keep2
            else:
                keep = keep1 & keep2

            # maybe exclude certain leafs
            if exclude_leafs is not None:
                keep[exclude_leafs] = False

            # keep track of kept idxes
            kept_idxs.extend(np.argwhere(keep)[:, 0].tolist())

            # now restrict to leaves of interes
            y_subset = y[keep, :]

            # plot
            plt.scatter(
                y_subset[:, 0], y_subset[:, 1],
                c=np.array(VisConfigs.CATEG_COLORS[cls])[None, :] / 255.,
                alpha=alphas[clno], s=point_size, edgecolors='none')

            point_size = point_size_ds * point_size

        plt.xlim(self._e0min, self._e0max)
        plt.ylim(self._e1min, self._e1max)
        plt.title(
            f'DTALE decisions ({postfix})', fontsize=14, fontweight='bold')
        # plt.show()
        # plt.savefig(opj(savedir, f'dectreeCol{postfix}.svg'))
        plt.savefig(opj(savedir, f'dectreeCol{postfix}.png'))

        return kept_idxs

    def save_and_plot_optimized_decision_paths(self):
        """
        Use different metrics to emphasize different things learned:
        - F1 score: typical case (most tumor nuclei in the dataset)
          VERSUS ...
        - precision: most discriminative case (textbook examples).
        Using F-1 helps us find nodes in our decision tree that correlate to
        the process used by the NuCLS model when making its "average"
        decision, whereas using the precision score allows us to understand
        when does the model decide that it's "sure" something is, say, a
        tumor nucleus.
        """
        for metric in ['F1', 'precision']:

            print(f'  Optimized for {metric}')

            # for each class, find the cluster (node) which best fits/explains
            # predictions from the NuCLS model (determined by metric of choice)
            best_nodes, best_stats = \
                self._get_best_node_for_each_class(metric=metric)

            # save decision tree traces for relevant classes
            kwargs = {
                'best_nodes': best_nodes,
                'postfix': f'_OptimizedFor{metric}',
            }
            self.save_dectree_traces(best_stats=best_stats, **kwargs)

            # visualize tree
            self.visualize_decision_tree_nodes(**kwargs)

            # color points associated with the best node for each class
            _ = self.visualize_decision_tree_classes(**kwargs)

    def plot_step_by_step_paths(self):

        # read precision traces
        with open(
            opj(self.savedir, f'dectree_traces_OptimizedForprecision.pkl'), 'rb') as f:  # noqa
            traces = pickle.load(f)

        # for each class, plot one node at a time, excluding downstream nodes
        for cls in self.classes_list:
            savedir = opj(self.savedir, cls)
            maybe_mkdir(savedir)
            classes_list = [cls]
            exclude_idxs = []
            for nix, node in enumerate(traces['nodes'][cls][::-1]):
                excl = self.visualize_decision_tree_classes(
                    best_nodes={cls: node},
                    classes_list=classes_list,
                    restrict_to_pcateg=True,
                    exclude_leafs=exclude_idxs,
                    savedir=savedir,
                    postfix=f'_{cls}_nodeidx-{nix}({node})',
                )
                exclude_idxs.extend(excl)

    def run_sequence(self):
        """Main workflow."""

        print('DTALE: Fitting model ...')
        self.fit_model()
        self.apply_model()

        print('DTALE: Parsing tree ...')
        self.set_leafs_for_all_subtrees()
        # self.set_node_tally()

        print('DTALE: Saving and plotting optimized decision paths ...')
        self.save_and_plot_optimized_decision_paths()
        self.plot_step_by_step_paths()
