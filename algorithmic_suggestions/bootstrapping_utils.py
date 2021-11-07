import os
import numpy as np
import cv2
from skimage.filters import threshold_otsu

import matplotlib
matplotlib.use('agg')
import matplotlib.pylab as plt
from matplotlib.colors import ListedColormap


def get_nuclei_from_region_prior(region_prior, htx, do_threshold= True):
    """
    given a region prior and hematoxylin channel, gets 
    nucleus segmentation
    args:
        region_prior - ndarray, 0,1 binary segmentation 
        htx - hematoxylin from deconvolution, np array (continuous)
    """
    soft_segmentation = region_prior * htx
    nuclei_soft = (region_prior * soft_segmentation) / 255
    
    if do_threshold:
        try:
            thresh = threshold_otsu(soft_segmentation[soft_segmentation > 0])
        except ValueError: # all values are zero
            thresh = 0
        nuclei_hard = region_prior * (soft_segmentation > thresh)
        return nuclei_soft, nuclei_hard
    else:
        return nuclei_soft


# noinspection PyUnresolvedReferences
def FC_CRF(pred, im, NLABELS, RESIZE_FACTOR= 5,
           DELTA_COL= 5, DELTA_SIZE= 5, n_steps=10,
           CERTAINTY= 0.7, mode= "multilabel"):
    """
    Fully Connected Conditional Random Fields
    See: 
    1- https://github.com/lucasb-eyer/pydensecrf (./examples/Non RGB Example.ipynb)
    2- http://warmspringwinds.github.io/tensorflow/tf-slim/2016/12/18/ ...
       image-segmentation-with-tensorflow-using-cnns-and-conditional-random-fields/

    Add (non-RGB) pairwise term 
    For example, in image processing, a popular pairwise relationship is the "bilateral" one, 
    which roughly says that pixels with either a similar color or a similar location are likely to 
    belong to the same class.
    """
    import pydensecrf.densecrf as dcrf
    from pydensecrf.utils import (
        unary_from_labels, unary_from_softmax,
        create_pairwise_bilateral, create_pairwise_gaussian)

    assert mode in ["multilabel", "prob"]

    (H0,W0) = im.shape[:2]
    H, W = (H0 // RESIZE_FACTOR, W0 // RESIZE_FACTOR)

    # regions RGB -- resize for efficiency
    image = cv2.resize(im, (W, H))

    # region labels -- note that cv2 is the only library I found
    # that downsizes without changing range of labels (which is 
    # obviously important for segmentation masks where values have meaning)
    labels0 = cv2.resize(pred, (W, H), interpolation=cv2.INTER_NEAREST)
    labels0 = (np.array(labels0)).astype('uint8')

    # The input should be the negative of the logarithm of probability values
    # Look up the definition of the softmax_to_unary for more information
    unary = unary_from_labels(
        labels= labels0, n_labels=NLABELS-1, 
        gt_prob= CERTAINTY, zero_unsure=True)

    # The inputs should be C-continious -- we are using Cython wrapper
    unary = np.ascontiguousarray(unary)
    d = dcrf.DenseCRF(H * W, NLABELS-1)
    d.setUnaryEnergy(unary)

    # NOTE: THE ORDER OF THE FOLLOWING TWO OPS
    # (create_pairwise_bilateral and create_pairwise_gaussian)
    # MAKES A DIFFERENCE !!!

    # This creates the color-dependent features --
    # because the segmentation that we get from CNN are too coarse
    # and we can use local color features to refine them
    feats = create_pairwise_bilateral(
        sdims=(DELTA_COL, DELTA_COL), schan=(20, 20, 20),
        img=image, chdim=2)
    d.addPairwiseEnergy(feats, compat=10,
                         kernel=dcrf.DIAG_KERNEL,
                         normalization=dcrf.NORMALIZE_SYMMETRIC)

    # This potential penalizes small pieces of segmentation that are
    # spatially isolated -- enforces more spatially consistent segmentations
    feats = create_pairwise_gaussian(
        sdims=(DELTA_SIZE, DELTA_SIZE), shape=(H, W))
    d.addPairwiseEnergy(feats, compat=3,
                        kernel=dcrf.DIAG_KERNEL,
                        normalization=dcrf.NORMALIZE_SYMMETRIC)

    # do inference
    Q = d.inference(n_steps)
    res = 1 + np.argmax(Q, axis=0).reshape((H, W))

    # resize back up to original
    res = cv2.resize(res, (W0, H0), interpolation=cv2.INTER_NEAREST)
    res = (np.array(res)).astype('uint8')
    return res


def occupy_full_GT_range(pred, n_classes):
    """occupy full GT range"""
    pred[0:n_classes,0] = np.arange(n_classes)
    return pred


def create_dir(folderpath):
    if not os.path.exists(folderpath):
        os.makedirs(folderpath)
    else:
        print("folder %s already exists! moving on." % (folderpath))


def get_shuffled_cmap(pltcmap= plt.cm.viridis, ncolors= 10000):
    """
    A random colormap for matplotlib -- shuffle viridis cmap
    see: https://gist.github.com/jgomezdans/402500
    """
    vals = np.linspace(0,1,ncolors)
    np.random.shuffle(vals)
    cmap_vals = pltcmap(vals)
    cmap_vals[0,...] = [0, 0, 0, 1]
    cmap = ListedColormap(cmap_vals)
    return cmap


def visualize_nuclei(im, instance_mask, labels_mask, cmaps, 
                    alpha_im=0.7, alpha_nuclei=1.0,
                    figsize=(15,15), savename= None, show= True):
    """
    visualize instance and mask information
    """
    def _show_cls(classlbl=1, cmap_to_use= plt.cm.viridis):
        # mask background so you can overlay
        this_cls = np.ma.masked_where(
            labels_mask != classlbl, instance_mask)
        # now show
        plt.imshow(
            this_cls, alpha=alpha_nuclei, interpolation=None, cmap= cmap_to_use)
    
    # ground work
    unique_nuclei_labels = np.unique(labels_mask)
    
    plt.figure(figsize=figsize)
    plt.imshow(im, alpha=alpha_im)
    for cno in unique_nuclei_labels:
        if cno == 0:
            continue
        _show_cls(classlbl = cno, cmap_to_use = cmaps[cno])
    if savename:
        plt.savefig(savename, format='tif', dpi=300, bbox_inches='tight')
    if show:
        plt.show()
