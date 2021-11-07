import numpy as np
from GeneralUtils import reverse_dict
from skimage.measure import find_contours
import random
import string


CODE_MAP = {
    "background":  0, 
    "tumor": 1, 
    "stroma": 2, 
    "lymphocyte": 3,
    "plasma_cells": 4,
    "other_immune": 5,
    "other": 6,
}
CODE_MAP_REVERSE = reverse_dict(CODE_MAP)


def convert_mask_to_three_channels(mask_loaded, class_ids, scores):
    """
    "Compress" mask into two channels.
    Args:
        mask_loaded - [M, N, n_instances], output from maskrcnn, 
                      where each channel belongsto one instance
        class_ids - [n_instances,] array-like. label of each instance
        scores - [n_instance,] array-like. confidence of each instance
    Returns:
        [M, N, 3] mask, where first channel encodes label, second 
        encodes instance membership, third encodes instance confidence
    """
    # extract pixelwise instance and label membership
    instance_info = np.argwhere(mask_loaded)
    instance_membership = instance_info[..., -1]
    instance_labels = np.zeros(instance_membership.shape)
    instance_confidence = np.zeros(instance_membership.shape)

    unique_labels = np.unique(class_ids)
    for lbl in unique_labels:
        lbl_instances = np.argwhere(class_ids == lbl)[:,0]
        instance_labels[np.isin(instance_membership, lbl_instances)] = lbl
        
    unique_confidence = np.unique(scores)
    for cnf in unique_confidence:
        cnf_instances = np.argwhere(scores == cnf)[:,0]
        instance_confidence[np.isin(instance_membership, cnf_instances)] = cnf

    # assign to mask, composed of:
    # label channel (0), instance channel (1), instance confidence channel (2)
    mask = np.zeros(mask_loaded.shape[:2] + (3,))
    mask[instance_info[:, 0], instance_info[:, 1], 0] = instance_labels
    mask[instance_info[:, 0], instance_info[:, 1], 1] = instance_membership
    mask[instance_info[:, 0], instance_info[:, 1], 2] = instance_confidence
    
    # save as float 32 because we want confidence to be preserved
    return mask.astype(np.float32)


def discard_edge_nuclei(mask, edge= 64, keep_threshold = 0.5):
    """
    Gets rid of nuclei predictions at the edge of the tile
    (i.e. keep nuclei if they more than keep_threshold 
     of their area is present centrally)
    Args:
        mask - output from convert_mask_to_two_channels, i.e. two-channel
               mask where channel 0 is labels and channel 1 is instances
    """
    # central pixels
    instances_central = mask[..., 1].copy()
    instances_central[:edge, :] = 0
    instances_central[-edge:, :] = 0
    instances_central[:, :edge] = 0
    instances_central[:, -edge:] = 0
    
    # edge pixels
    instances_edges = mask[..., 1].copy()
    instances_edges[edge:-edge, edge:-edge] = 0

    # unique instances in center and edge
    unique_central = np.unique(instances_central)
    unique_edges = np.unique(instances_edges)

    # restrict to unique instances that touch edge
    unique_central = list(unique_central[np.isin(unique_central, unique_edges)])
    unique_central.remove(0)

    # keep central nuclei if they more than keep_threshold
    # of their area is present centrally
    for nucleus_id in unique_central:
        nucleus_count_central = np.sum(instances_central == nucleus_id)
        nucleus_count_edges = np.sum(instances_edges == nucleus_id)
        if (nucleus_count_central / (nucleus_count_central+ nucleus_count_edges)) > keep_threshold:
            instances_central[instances_edges == nucleus_id] = nucleus_id
        else:
            instances_central[instances_central == nucleus_id] = 0

    discard = np.argwhere(instances_central == 0)
    mask[discard[:, 0], discard[:, 1], :] = 0

    return mask


def add_contour_channel(mask):
    """
    Adds a contour channel that is aware of instances, 
    i.e. pixel value indicates instance membership.    
    Args:
        mask - output from convert_mask_to_three_channels, i.e. three-channel
               mask where channel 0 is labels, channel 1 is instances, 
               and channel 2 is instance confidence
    Output:
        same but with a last channel added containing instance contours
    """
    # initialize channel to save instance-aware contours
    mask = np.concatenate((mask, np.zeros(mask.shape[:2] + (1,))), axis=-1)

    # go through instances and get contours
    unique_instances = list(np.unique(mask[..., 1]))
    unique_instances.remove(0)

    for instanceid in unique_instances:

        mask_instance = np.int32(mask[..., 1] == instanceid)
        contours = find_contours(mask_instance, 0.5)

        # if no contours, ignore instance
        if len(contours) < 1:
            continue

        # some contours may be junk (especially if processing NN output)
        # so only restrict to main instance contour
        longest_contour = np.argmax([len(j) for j in contours])
        
        # if longest contour is < 10 pixels, this is an artifact
        if len(contours[longest_contour]) < 10:
            continue
        
        # now assign
        contours = np.int32(contours[longest_contour])

        # assign contour pixels to label channel
        pos = np.argwhere(mask_instance)[0]
        mask[contours[:, 0], contours[:, 1], 0] = mask[pos[0], pos[1], 0]
        
        # assign contour pixels to segmentation channel
        mask[contours[:, 0], contours[:, 1], 1] = instanceid
        
        # assign contour pixels to confidence channel
        mask[contours[:, 0], contours[:, 1], 2] = mask[pos[0], pos[1], 2]
        
        # assign contour pixels to contour channel
        mask[contours[:, 0], contours[:, 1], -1] = instanceid

    return mask


def add_nucleus_info_to_df(Annots_DF, iminfo, mask):
    """
    Add nucleus instance from an FOV information to annotation dataframe.
    Args:
        Annots_DF: Annotation dataframe
        iminfo: dataset.image_info from maskrcnn dataset class
        mask: 4 channels, in order: labels, instances (solid), 
              instance_confidence, instance_contours
    """
    # initialize position
    dfpos = Annots_DF.shape[0]

    # Get info that applies to all instances in fov
    # remember, y is columns, x is rows

    slide_name = iminfo['id'].split("_")[0]

    roi_xmin = int(iminfo['id'].split("_xmin")[1].split("_")[0])
    roi_ymin = int(iminfo['id'].split("_ymin")[1].split("_")[0])
    roi_offset_xmin_ymin = "%d,%d" % (roi_xmin, roi_ymin)

    fov_xmin = int(iminfo['id'].split("_colmin")[1].split("_")[0])
    fov_ymin = int(iminfo['id'].split("_rowmin")[1].split("_")[0])
    fov_offset_xmin_ymin = "%d,%d" % (fov_xmin, fov_ymin)

    x_offset = roi_xmin + fov_xmin
    y_offset = roi_ymin + fov_ymin

    # go through nuclei instances and add info to df
    unique_instances = list(np.unique(mask[..., 3]))
    unique_instances.remove(0)

    # handle if no instances -- return df as is
    if len(unique_instances) < 1:
        return Annots_DF

    for instanceid in unique_instances:

        # pixel locations for this instance
        # remember, y is columns, x is rows
        pxlocs = np.argwhere(mask[..., 3] == instanceid).astype(np.int32)
        ymin, xmin = np.min(pxlocs, axis= 0)
        ymax, xmax = np.max(pxlocs, axis= 0)
        center_x = (xmin+xmax)//2 + x_offset
        center_y = (ymin+ymax)//2 + y_offset

        # unique ID
        gibbrish = ''.join(random.choice(
            string.ascii_uppercase + string.digits) for _ in range(5))
        unique_nucleus_id = "%s_%d_%d_%s" % (slide_name, center_x, center_y, gibbrish)

        
        # add info
        Annots_DF.loc[dfpos, "unique_nucleus_id"] = unique_nucleus_id
        Annots_DF.loc[dfpos, "slide_name"] = slide_name
        Annots_DF.loc[dfpos, "nucleus_label"] = CODE_MAP_REVERSE[int(mask[pxlocs[0,0], pxlocs[0,1], 0])]
        Annots_DF.loc[dfpos, "nucleus_label_confidence"] = int(
            mask[pxlocs[0,0], pxlocs[0,1], 2] * 100) # save as % for SQL compatibility
        Annots_DF.loc[dfpos, "fov_offset_xmin_ymin"] = fov_offset_xmin_ymin
        Annots_DF.loc[dfpos, "roi_offset_xmin_ymin"] = roi_offset_xmin_ymin
        Annots_DF.loc[dfpos, "center_relative_to_slide_x_y"] = "%d,%d" % (center_x, center_y)
        Annots_DF.loc[dfpos, "bounding_box_relative_to_slide_xmin_ymin_xmax_ymax"] = \
            "%d,%d,%d,%d" % (xmin+x_offset, ymin+y_offset, xmax+x_offset, ymax+y_offset)
        Annots_DF.loc[dfpos, "boundary_relative_to_slide_x_coords"] = \
            ",".join([str(j) for j in pxlocs[:,1]+x_offset])
        Annots_DF.loc[dfpos, "boundary_relative_to_slide_y_coords"] = \
            ",".join([str(j) for j in pxlocs[:,0]+y_offset])

        # increment
        dfpos += 1
    
    return Annots_DF
