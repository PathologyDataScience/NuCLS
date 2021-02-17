import matplotlib.pylab as plt
import numpy as np
from scipy.cluster.hierarchy import dendrogram
from sklearn.cluster import AgglomerativeClustering


class ConstrainedAgglomerativeClustering(object):

    def __init__(
            self, linkage_thresh, linkage='complete',
            affinity='precomputed', dontlink=None):
        """Constrained AgglomerativeClustering.

        Current constrains allowed is "dont link", preventing certain
        leafs from being clumped together in the same node. This is
        NOT optimized for efficiency. It is meant to do an OK job with least
        effort as part of a bigger project.In order words, only use this
        for with moderate numbers of samples.

        The way this works is as follows:
        ---------------------------------------------------------------
        1 Do the hierarchical clustering and threshold,
        2 For each cluster Ci (corresponding to top-level node Ni)
         2.1 For each "dont-link" set Sj
          2.1.1 Check if more than one member in Sj is present in Ci
          2.1.2 For each extra member Sjk
           2.1.2.1 Check the next low-level node Ni-1
           - If there are no members from Sj in Ni-1
             - If Ci-1 does not exist, set Ni-1 as a new cluster Ci-1
             - Assign Sjk to Ci-1
           - Else check the next low-level node Ni-2 (repeat 2.1.2.1)
           2.1.2.2 If no nodes without members from Sj found
             - Assign Sjk as a separate one-leaf cluster
        ---------------------------------------------------------------

        **Note:**
        Simply using the "connectivity constraint" in scipy fails because:
        1. its unstable in some situations
        2. members in one "dont link" group could be indirectly connected
        through other members from other don't link groups

        Parameters
        ----------
        linkage_thresh : float
            linkage threshold

        linkage : str
            see sklearn.cluster.AgglomerativeClustering

        affinity : str
            see sklearn.cluster.AgglomerativeClustering

        dontlink : list
            each entry is a list of leafs that should not be together
            in the same cluster

        """
        # assign
        self.linkage = linkage
        self.linkage_thresh = linkage_thresh
        self.affinity = affinity

        # list of lists, where each list contains leafs that should
        # not be together in the same cluster
        self.dontlink = [] if dontlink is None else dontlink

        # Init Agglomerative Clustering model
        self.model = AgglomerativeClustering(
            linkage=self.linkage, affinity=self.affinity, n_clusters=None,
            distance_threshold=0,  # to have full dendrogram
            # distance_threshold=self.linkage_thresh,
        )

    # -------------------------------------------------------------------------

    def run(self, cost):
        """Fit then fix the clusters obtained from model to be constrained."""
        self.fit_unconstrained_model(cost=cost)
        self._set_flattened_node_subtrees()
        self._set_subtree_within_linkage_threshold()
        self._set_clusters()

    # -------------------------------------------------------------------------

    def fit_unconstrained_model(self, cost):
        """Fit unconstrained model using AgglomerativeClustering.fit.

        Parameters
        ----------
        cost : numpy.array
            cost matrix. See sklearn.cluster.AgglomerativeClustering.fit

        """
        self.model.fit(cost)

        # for convenience
        self.children_ = self.model.children_
        self.labels_ = self.model.labels_
        self.n_samples = len(self.labels_)
        self.distances_ = self.model.distances_

    # -------------------------------------------------------------------------

    def plot_dendrogram(self, **kwargs):
        """Create linkage matrix and then plot the dendrogram.

        Source: https://scikit-learn.org/stable/auto_examples/cluster/ ...
        ... plot_agglomerative_dendrogram.html#sphx-glr-auto-examples ...
        ... -cluster-plot-agglomerative-dendrogram-py
        """
        # create the counts of samples under each node
        counts = np.zeros(self.children_.shape[0])
        for i, merge in enumerate(self.children_):
            current_count = 0
            for child_idx in merge:
                if child_idx < self.n_samples:
                    current_count += 1  # leaf node
                else:
                    current_count += counts[child_idx - self.n_samples]
            counts[i] = current_count

        linkage_matrix = np.column_stack([self.children_, self.distances_,
                                          counts]).astype(float)

        # Plot the corresponding dendrogram
        dendrogram(linkage_matrix, **kwargs)

    def plot_dendrogram_for_model(self, p=5, show_thresh=True):
        plt.figure(figsize=(7, 7))
        plt.title('Hierarchical Clustering Dendrogram - Top %d levels' % p)
        self.plot_dendrogram(truncate_mode='level', p=p)
        plt.xlabel("Leaf index or (Number of leafs in node)")
        plt.ylabel("Linkage")
        if show_thresh:
            plt.axhline(self.linkage_thresh, linestyle='--', c='r')
        plt.show()

    # -------------------------------------------------------------------------

    def _set_flattened_node_subtrees(self):
        """Get all the leafs and subnodes within a node for all nodes.

        Sets the following properties ..

        self.flattened - dict, indexed by node.
            Each entry has these keys
            - nodes - list, all subnodes
            - leafs - list, all subleafs

        self.leaf_parents - dict, indexed by leaf.
            Each entry is a list of the leaf's parents from the bottom up.

        """
        # From the documentation:
        # children_ : array-like of shape (n_samples-1, 2)
        #     The children of each non-leaf node. Values less than `n_samples`
        #     correspond to leaves of the tree which are the original samples.
        #     A node `i` greater than or equal to `n_samples` is a non-leaf
        #     node and has children `children_[i - n_samples]`. Alternatively
        #     at the i-th iteration, children[i][0] and children[i][1]
        #     are merged to form node `n_samples + i`
        #
        # **Note:**
        # it is important for the unique nodes to be sorted in ascending
        # fashionto make sure we traverse from the bottom up and therefore
        # benefit from dynamic programming
        #
        self.root_node = self.children_.shape[0] + self.n_samples - 1
        self.unique_nodes = list(range(self.n_samples, self.root_node + 1))
        self.flattened = {
            nd: {"nodes": [], "leafs": []} for nd in self.unique_nodes}

        # this will hold the parents of a leaf from the bottom up
        self.leafs = list(range(self.n_samples))
        self.leaf_parents = {lf: [] for lf in self.leafs}

        # Now traverse
        for node in self.unique_nodes:
            children = self.children_[node - self.n_samples]
            for child in children:
                if child < self.n_samples:  # this is a leaf
                    self.flattened[node]["leafs"].append(child)
                else:
                    self.flattened[node]["nodes"].append(child)
                    self.flattened[node]["nodes"].extend(
                        self.flattened[child]["nodes"])
                    self.flattened[node]["leafs"].extend(
                        self.flattened[child]["leafs"])
                    for lf in self.flattened[child]["leafs"]:
                        self.leaf_parents[lf].append(child)

    # -------------------------------------------------------------------------

    def _set_subtree_within_linkage_threshold(self):
        """Find subtree within the linkage threshold.

        Sets the following properties ..

        self.subtree - dict, with the following keys
        - nodes - list, all nodes in the subtree (i.e. within threshold)
        - leafs - list, all leafs in the subtree (i.e. within threshold)
        - top_nodes - list, top-level nodes within the subtree
        - top_leafs - list, leafs that not part of any node within the subtree

        """
        # The number of "cuts" to the linkage threshold line is
        #   = number of nodes ABOVE the thresholf line + 1
        # The following implementation is copied as is from
        # sklearn.cluster.AgglomerativeClustering.fit
        self.n_clusters = np.count_nonzero(
            self.distances_ >= self.linkage_thresh) + 1

        # these are all the nodes ABOVE the linkage threshold
        nodes_above_thresh = np.arange(
            self.root_node - self.n_clusters + 1, self.root_node + 1, 1)
        lowest_node_above_thresh = np.min(nodes_above_thresh)

        # find top-level nodes + leafs below the linkage threshold
        # these are nodes or leafs that appear in the children of nodes
        # above the linkage threshold
        end = np.argmax(self.distances_ > self.linkage_thresh)
        cuts = np.unique(self.children_[end:, :].ravel())
        cuts = cuts[cuts < lowest_node_above_thresh]

        # now assign
        self.subtree = {
            "nodes": np.arange(self.n_samples, lowest_node_above_thresh),
            "leafs": np.arange(0, self.n_samples),
            "top_nodes": cuts[cuts >= self.n_samples].tolist(),
            "top_leafs": cuts[cuts < self.n_samples].tolist(),
        }

    # -------------------------------------------------------------------------

    def _set_clusters(self):
        """Get the contrained clusters.

        Sets the following properties ..

        self.clusters - dict, indexed by node/leaf
            Each entry is a list of the leafs within the cluster.

        """
        self.clusters = {
            nd: self.flattened[nd]["leafs"]
            for nd in self.subtree["top_nodes"]}
        self.clusters.update({lf: [lf] for lf in self.subtree["top_leafs"]})

        # find nodes (anchors) with more than one match from the same user
        for node in self.subtree["top_nodes"]:
            for nolinksubset in self.dontlink:
                cleafs = np.array(self.flattened[node]["leafs"])
                cleafs_user = list(cleafs[np.in1d(cleafs, nolinksubset)])

                if len(cleafs_user) == 0:
                    continue

                # first is kept in this cluster
                cleafs_user.pop(0)

                # "Demote" all remaining user elements, if any, to lower level
                for cleaf in cleafs_user:

                    # remove from this top cluster
                    self.clusters[node].remove(cleaf)

                    # Go through parent nodes from top to bottom
                    # "top" being defined as just below linkage threshold
                    end = self.leaf_parents[cleaf].index(node)
                    clparents = self.leaf_parents[cleaf][:end]
                    assigned = False
                    for clparent in clparents[::-1]:
                        if not assigned:
                            # Case 1: the parent is not a top node; create it
                            if clparent not in self.clusters.keys():
                                self.clusters[clparent] = [cleaf]
                                assigned = True
                            # Case 2: the parent is a top node and doesn't
                            # have any of this users' leafs
                            elif len(set(self.clusters[clparent]).intersection(
                                    set(cleafs_user))) == 0:
                                self.clusters[clparent].append(cleaf)
                                assigned = True

                    # Case 3: There are no parents left to assign leaf because
                    #   a. there are none or
                    #   b. all have at a leaf from the same user
                    if not assigned:
                        self.clusters[cleaf] = [cleaf]

    # -------------------------------------------------------------------------
