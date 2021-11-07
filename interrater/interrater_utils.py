import io
import copy
import os
from os.path import join as opj
from PIL import Image
from sqlalchemy import create_engine
import matplotlib.pylab as plt
from matplotlib import patches
from matplotlib.colors import ListedColormap
from pandas import read_sql_query
from pandas import DataFrame, concat, Series
import numpy as np
from PIL import ImageDraw
from collections import Counter
from typing import Tuple, List, Dict, Any
from histomicstk.annotations_and_masks.annotation_and_mask_utils import \
    np_vec_no_jit_iou, get_scale_factor_and_appendStr, \
    get_image_from_htk_response
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.preprocessing import label_binarize
from copy import deepcopy

from configs.nucleus_style_defaults import Interrater as ir, \
    DefaultAnnotationStyles, NucleusCategories  # noqa
from interrater.constrained_agglomerative_clustering import \
    ConstrainedAgglomerativeClustering  # noqa
from GeneralUtils import calculate_4x4_statistics


def _maybe_mkdir(folder):
    os.makedirs(folder, exist_ok=True)


def _get_color_from_rgbstr(rgbstr):
    return [int(j) / 255 for j in rgbstr[4:-1].split(',')]


def _add_fovtype_to_eval_fov(fovinfo, dbcon):
    """Add fov type to fovinfo dict."""
    # any subset and user will work
    evname = list(fovinfo.keys())[0]
    userfov = fovinfo[evname][list(fovinfo[evname].keys())[0]]
    contdf = read_sql_query(f"""
    SELECT "element_girder_id", "group", 
           "xmin", "ymin", "xmax", "ymax"
    FROM "annotation_elements"
    WHERE "fov_id" = {userfov['fov_id']}
    ;""", dbcon)

    # get fov group and add
    grps = list(contdf.loc[:, "group"])
    fovinfo['fov_type'] = [j for j in grps if "fov" in j.lower()][0]

    return fovinfo


def get_fovinfos_for_interrater(dbcon):
    """Get fov ids for inter-rater analysis.

    Get dict which is indexed by the FOV name and contains five entries
    (EVAL_SET_1, EVAL_SET_2, EVAL_SET_3, EVAL_SET_3), containing the
    information about the annotations of THIS fov under different conditions:

    - U-control: the observers place bounding boxes around all nuclei
                 and they are not biased by any pre-existing boundaries.
    - B-control: Existing nucleus bounds are obtained using "traditional"
                image processing algorithms (smoothing, thresholding, connected
                components etc). These are then corrected by the participants
                the study's protocol (dots inside correct polygon bounds having
                and bounding boxes around nuclei without correct bounds.
    - E: Main evaluation set. This is the same as B-control but the bounds
                are obtained from traditional methods are used to train
                mask-RCNN to obtain more accurate bounds and labels. This is
                also the method that is used for obtaining the core set.
    - BT-control: Same as E but instead of obtaining the labels to the
                nuclei used to train the mask-RCNN by assigning them the same
                label as the region labels manually annotated, here the region
                labels are obtained by quickly dialing knobs in the HSI space.
    - fov_type: See Roche patent. This has to do with how the FOV location
                itself was obtained.

    Each entry is also a dict indexed by the name of the observer and has the
    fov_id for the annotations by that user in that fov location.
    """
    fovinfos = dict()

    for evalset in ir.EVALSET_NAMES:

        # alias = ir.REVERSE_ANNSETS[evalset]['alias']
        alias = evalset

        # find fovs for this evaludation set
        fovs_thisset = read_sql_query(f"""
        SELECT "fov_id", "fovname", "QC_label", "slide_name",
               "sf", "maybe-xmin", "maybe-ymin", "maybe-xmax", "maybe-ymax",
               "XMIN", "YMIN"
        FROM "fov_meta"
        WHERE "slide_id" IN (
            SELECT DISTINCT "itemId" 
            FROM "annotation_docs"
            WHERE "subset" = "{evalset}"
        )
        ;""", dbcon)

        for _, fov in fovs_thisset.iterrows():

            fov = fov.to_dict()

            # don't include FOVs that are not done
            if fov['QC_label'] != 'COMPLETE':
                continue

            # Note that we CANNOT use the locations in the fov name
            # as these are slightly expanded beyond the FOV boundary to include
            # overflowing annotations. Instead we use the fov locations from
            # fov meta. Alternatively, we could have read the FOV contours csv
            # file and mapped it to slide coordinates
            observer, _, tmp = fov['fovname'].split('_#_')
            sldname, tmp = tmp.split("_id")
            standard_fovname = "%s_left-%d_top-%d_bottom-%d_right-%d" % (
                sldname, fov['maybe-xmin'], fov['maybe-ymin'],
                fov['maybe-ymax'], fov['maybe-xmax'],
            )

            if standard_fovname not in fovinfos.keys():
                fovinfos[standard_fovname] = {alias: dict()}
            elif alias not in fovinfos[standard_fovname]:
                fovinfos[standard_fovname][alias] = dict()

            # assign
            fovinfos[standard_fovname][alias][observer] = fov

    for _, fovinfo in fovinfos.items():
        _add_fovtype_to_eval_fov(fovinfo, dbcon=dbcon)

    return fovinfos


def _convert_coordstr_to_absolute(coordstr: str, sf: float, minc: int) -> str:
    return ','.join([
        str(int((int(c) / sf) + minc)) for c in coordstr.split(',')])


def _modify_bbox_coords_to_base_mag(contdf, userfov):
    """Modify bbox coordinates to be at base slide magnification."""
    # bounding box
    for locstr in ['xmin', 'xmax']:
        contdf.loc[:, locstr] = np.int32(
            contdf.loc[:, locstr] / userfov['sf'] + userfov['XMIN'])
    for locstr in ['ymin', 'ymax']:
        contdf.loc[:, locstr] = np.int32(
            contdf.loc[:, locstr] / userfov['sf'] + userfov['YMIN'])

    # actual coordinates
    for c in ('x', 'y'):
        contdf.loc[:, f'coords_{c}'] = contdf.loc[:, f'coords_{c}'].apply(
            lambda cs: _convert_coordstr_to_absolute(
                cs, sf=userfov['sf'], minc=userfov[f'{c.upper()}MIN']))

    return contdf


def _get_all_contours_for_eval_fov(fovinfo, dbcon, max_bbox_side=100):
    """Get all contours for this FOV from various users."""
    all_conts = DataFrame()
    fov_conts = DataFrame()

    for evalset in ir.EVALSET_NAMES:

        if evalset not in fovinfo.keys():
            continue

        finfo = fovinfo[evalset]

        for user, userfov in finfo.items():

            contdf = read_sql_query(f"""
            SELECT "element_girder_id", "group", 
                   "xmin", "ymin", "xmax", "ymax", "coords_x", "coords_y"
            FROM "annotation_elements"
            WHERE "fov_id" = {userfov['fov_id']}
            ;""", dbcon)

            # make sure its at base magnification and has no offset
            contdf = _modify_bbox_coords_to_base_mag(
                contdf=contdf, userfov=userfov)
            contdf.loc[:, "user"] = user
            contdf.loc[:, "evalset"] = evalset

            # get fov group and remove from dataframe
            grps = list(contdf.loc[:, "group"])
            fovels = [
                (i, j) for i, j in enumerate(grps) if "fov" in j.lower()]
            fovcont = contdf.loc[[j[0] for j in fovels], :]
            for elid, _ in fovels:
                contdf.drop(elid, axis=0, inplace=True)
            contdf.loc[:, "fov_type"] = fovels[0][1]

            # now we add to main dict
            all_conts = concat(
                (all_conts, contdf), axis=0, sort=False, ignore_index=True)
            fov_conts = concat(
                (fov_conts, fovcont), axis=0, sort=False, ignore_index=True)

    all_conts.index = all_conts.loc[:, "element_girder_id"]
    fov_conts.index = fov_conts.loc[:, "element_girder_id"]

    # get rid of "nuclei" which are really FOV annotations that were
    # mistakenly clicked by the observer and changed to a non-fov label
    right_width = (
        all_conts.loc[:, "xmax"] - all_conts.loc[:, "xmin"]) < max_bbox_side
    right_height = (
        all_conts.loc[:, "ymax"] - all_conts.loc[:, "ymin"]) < max_bbox_side
    all_conts = all_conts.loc[right_width, :]
    all_conts = all_conts.loc[right_height, :]

    return all_conts, fov_conts


def _get_iou_for_fov(all_conts):
    """Get ious for all bboxes/bounds in fov.

    The target of the code below is to get a matrix comparing
    the IOU or each potential nucleus with all other potential nuclei.
    Note that a "potential" nucleus is defined as a bounding
    box created by one of the observers OR the bounding box of a polygon
    "approved" by one of the observers. We use the bounding box of approved
    polygons as opposed to the polygons themselves for two reasons:
    1. To avoid bias caused by artificially low IOU values when a polygon
       is used versus a bounding box. We don't want an apple to oranges
       comparison!
    2. Efficiency! Comparing bounding boxes is a vectorized operation that
       can be done for all bboxes in an FOV at once.

    """
    # get iou of bounding boxes
    sliced = np.array(
        all_conts.loc[:, ["xmin", "ymin", "xmax", "ymax"]], dtype=int)
    iou = np_vec_no_jit_iou(bboxes1=sliced, bboxes2=sliced)
    iou = DataFrame(iou, index=all_conts.index, columns=all_conts.index)
    return iou


def _get_clusters_for_fov(all_conts, iou, min_iou=0.5, constrained=True):
    """Agglomerative clustering of bounding boxes.

    Parameters
    ----------
    all_conts
    iou
    min_iou
    constrained

    Returns
    -------

    """
    # if constrained, find annotation indices that cannot appear in the
    # same cluster. In this case, annotations from the same user (and FOV)
    # CANNOT map to the same anchor since the user's intention, by definition,
    # is to annotate two separate nuclei
    dontlink = _get_annlocs_for_same_user(all_conts) if constrained else None

    # Hierarchical agglomerative clustering
    model = ConstrainedAgglomerativeClustering(
        linkage_thresh=1 - min_iou,
        linkage='complete', affinity='precomputed',
        dontlink=dontlink,
    )

    # now we fit the model
    model.run(cost=1 - np.array(iou.values, dtype=float))

    return model


def _get_relative_anchors(cluster_medoids, bounds):
    """Get medoid coords relative to fetched RGB."""
    relative_medoids = cluster_medoids.copy()
    relative_medoids.loc[:, "xmin"] -= bounds["XMIN"]
    relative_medoids.loc[:, "ymin"] -= bounds["YMIN"]
    relative_medoids.loc[:, "xmax"] -= bounds["XMIN"]
    relative_medoids.loc[:, "ymax"] -= bounds["YMIN"]
    relative_medoids = relative_medoids * bounds['sf']
    return relative_medoids.astype('int')


def _get_coords_from_coordstr(
        coordstr_x: str, coordstr_y: str) -> List[List[int]]:
    return [
        [int(j) for j in xy] for xy in
        zip(coordstr_x.split(','), coordstr_y.split(','))
    ]


def create_mask_from_coords(
        coords, min_x=None, min_y=None, max_x=None, max_y=None):
    """Create a binary mask from given vertices coordinates.

    Source: This is modified from code by Juan Carlos from David Gutman Lab.
    This version is modified from histomicstk.annotation_and_mask_utils

    Parameters
    -----------
    coords : np arrray
        must be in the form (e.g. ([x1,y1],[x2,y2],[x3,y3],.....,[xn,yn])),
        where xn and yn corresponds to the nth vertix coordinate.

    Returns
    --------
    np array
        binary mask

    """
    polygon = coords.copy()

    if any([j is None for j in [min_x, min_y, max_x, max_y]]):
        # use the smallest bounding region, calculated from vertices
        min_x, min_y = np.min(polygon, axis=0)
        max_x, max_y = np.max(polygon, axis=0)

    # get the new width and height
    width = int(max_x - min_x)
    height = int(max_y - min_y)
    # shift all vertices to account for location of the smallest bounding box
    polygon[:, 0] = polygon[:, 0] - min_x
    polygon[:, 1] = polygon[:, 1] - min_y

    # convert to tuple form for masking function (nd array does not work)
    vertices_tuple = tuple(map(tuple, polygon))
    # make the mask
    img = Image.new('L', (width, height), 0)
    ImageDraw.Draw(img).polygon(vertices_tuple, outline=1, fill=1)

    return np.array(img, dtype='int32')


def _get_anchor_from_single_cluster(
        relevant_contours, cluster_iou, usrs,
        manualusr='Ehab_Hafiz', manualusrgrp='SPs'):
    """Get the true nucleus location (aka "anchor") for a single cluster.

    Parameters
    ----------
    relevant_contours: DataFrame
    cluster_iou: DataFrame
        iou of elements that mapped to that cluster
    usrs: dict
        indexed by eval set name. Each entry is a dict, where keys
        are the names of participants, and the values are either
        "undetected" or "DidNotAnnotateFOV". This is used to initialize
        the anchor dict and to differentiate between those who annotated
        the FOV, and those who annotated it but did not detect the nucleus.
    manualusr: str

    Returns
    -------

    """
    manualusr = ir.PARTICIPANT_ALIASES[manualusr]

    def _get_anninfo_for_subgroup(evalset):
        """Get the info about annotations by participants in an evalset."""
        evdf = relevant_contours.loc[
            relevant_contours.loc[:, 'evalset'] == evalset, :]
        ev_anninfo = {
            who: {'annids': [], 'labels': [], 'users': [], 'polylines': []}
            for who in ir.who.keys()}

        for annid, row in evdf.iterrows():
            # Add label from user. Note that if we did not do constrained
            # clustering, and we ended up with more than one annotation
            # from the same user, this would simply overwrite the label
            # BUT, no worries, the count will be increased and the grder ID
            # will still be saved to the matches so that we can later show
            # the effect of constraining
            usrs[evalset][row['user']] = row['group']
            # Add girder ID and label from user to each relevant group
            for who, ppl in ir.who.items():
                if row['user'] in ppl:
                    ev_anninfo[who]['annids'].append(annid)
                    ev_anninfo[who]['labels'].append(row['group'])
                    ev_anninfo[who]['users'].append(row['user'])
                    if (evalset != 'U-control') and (
                            row['type'] == 'polyline'):
                        ev_anninfo[who]['polylines'].append(row['user'])
        return ev_anninfo

    def _get_medoid_using_iou(gids, justid=False):
        """Get medoid of a bunch of annotations using their girder id.
        NOTE: cluster "medoid" is defined as the element with maximum mean iou
        with all other elements in the cluster. In other words, it is
        most representative of this cluster. Note that the label for
        the medoid is irrelevant.
        """
        iousubset = cluster_iou.loc[gids, :]
        contsubset = relevant_contours.loc[gids, :]
        medoid_id = iousubset.index[np.argmax(np.array(iousubset.mean(1)))]
        if justid:
            return medoid_id
        else:
            md = contsubset.loc[medoid_id, ['xmin', 'ymin', 'xmax', 'ymax']]
            return dict(md)

    def _get_anchor_bbox(anninfo):
        """Get the cluster anchor (i.e. representative element). Here's the
        order of preference:
        1- If an unbiased manual segmentation boundary exists (by Ehab Hafiz),
           then we save it and use it for the anchor bounding box limits.
        2- We use pathologist annotations from the unbiased control set,
           if they exists in the cluster, to get the "proper" anchor location.
        3- We use NP annotations from the unbiased control set.
        4- We uss the medoid of all matched elements, regardless of set
        """
        # init anchor bbox location
        anchorbbox = {k: np.nan for k in ('xmin', 'ymin', 'xmax', 'ymax')}
        has_manual_bounary = manualusr in \
            anninfo['U-control'][manualusrgrp]['users']
        manual_boundary = None
        unbiased_pids = anninfo['U-control']['Ps']['annids']
        unbiased_npids = anninfo['U-control']['NPs']['annids']

        # First preference: Use manual boundary
        if has_manual_bounary:
            rows = relevant_contours.loc[anninfo['U-control'][manualusrgrp][
                'annids'], :]
            row = rows.loc[rows.loc[:, 'user'] == manualusr, :].iloc[0, :]
            manual_boundary = {
                'coords_x': row['coords_x'], 'coords_y': row['coords_y']}

            # bbox of manual boundary take precedence
            anchorbbox.update({k: row[k] for k in (anchorbbox.keys())})

        # Second preference: use pathologist annotations from the unbiased set
        elif len(unbiased_pids) > 0:
            anchorbbox.update(_get_medoid_using_iou(unbiased_pids))

        # Third preference: use NP annotations from the unbiased set
        elif len(unbiased_npids) > 0:
            anchorbbox.update(_get_medoid_using_iou(unbiased_npids))

        # Last preference: use all annotations
        else:
            anchorbbox.update(_get_medoid_using_iou(relevant_contours.index))

        return anchorbbox, manual_boundary

    def _get_MV_inferred_label_for_who(evalset, who):
        labels = anninfo[evalset][who]['labels']
        if len(labels) > 0:
            return Counter(labels).most_common()[0]
        return np.nan, np.nan

    def _get_algorithmic_boundary_for_evalset(evalset):
        """Get algorithmic boundary for nucleus.
        Remember that each eval set has its own algorithmic boundary,
        except of course the unbiased control set, which has none.
        """
        polylines = relevant_contours.loc[relevant_contours.loc[
            :, 'type'] == 'polyline', :]
        polylines = polylines.loc[polylines.loc[:, 'evalset'] == evalset, :]

        # unbiased controls don't have algorithmic boundaries
        if (polylines.shape[0] < 1) or (evalset == 'U-control'):
            return {
                f'algorithmic_coords_{c}': np.nan for c in ('x', 'y')}

        # find the medoid polyline. Note that even though everyone is presumed
        # to have clicked the same algorithmic bounday, it is possible that
        # multiple clicked algorithmic boundaries matched to the same cluster
        mid = _get_medoid_using_iou(polylines.index, justid=True)
        return {
            'algorithmic_coords_x': polylines.loc[mid, 'coords_x'],
            'algorithmic_coords_y': polylines.loc[mid, 'coords_y'],
        }

    def _get_anchor_dict(evalset):
        """Make sure all fields exist."""
        anchor_id = ",".join([
            str(anchorbbox[k]) for k in ('xmin', 'ymin', 'xmax', 'ymax')])
        anch = {
            'anchor_id': anchor_id,
            'fovname': np.nan,
            'min_iou': np.nan,
            'evalset': evalset,
        }
        anch.update(anchorbbox)
        anch.update({f'{loc}_relative': np.nan for loc in anchorbbox.keys()})
        anch.update({
            'has_manual_boundary': manual_boundary is not None,
            'has_algorithmic_boundary': any([
                len(anninfo[evalset][who]['polylines']) > 0
                for who in ir.who.keys()]),
            'algorithmic_vs_manual_intersect': np.nan,
            'algorithmic_vs_manual_sums': np.nan,
            'algorithmic_vs_manual_IOU': np.nan,
            'algorithmic_vs_manual_DICE': np.nan,
        })
        # no of matches per experience level
        anch.update({
            f'n_matches_{who}': float(len(anninfo[evalset][who]['annids']))
            for who in ir.who.keys()})
        anch.update({
            f'UNBIASED_n_matches_{who}': np.nan
            for who in ir.CONSENSUS_WHOS})
        # tally and no of algorithmic boundary approvals per experience level
        anch.update({
            f'algorithmic_clicks_{who}':
                ",".join(anninfo[evalset][who]['polylines'])
            for who in ir.who.keys()})
        anch.update({
            f'n_algorithmic_clicks_{who}': float(len(
                anninfo[evalset][who]['polylines']))
            for who in ir.who.keys()})
        # consensus label per experience level
        consensuses = dict()
        for who in ir.CONSENSUS_WHOS:  # ir.who.keys()
            # majority voting
            consensuses[f'MV_inferred_label_{who}'], \
                consensuses[f'MV_inferred_label_count_{who}'] = \
                _get_MV_inferred_label_for_who(evalset, who)
            consensuses[f'UNBIASED_MV_inferred_label_{who}'] = np.nan
            consensuses[f'UNBIASED_MV_inferred_label_count_{who}'] = np.nan
            # expectation maximization
            consensuses[f'EM_decision_boundary_is_correct_{who}'] = 0
            consensuses[f'EM_inferred_label_{who}'] = np.nan
            consensuses[f'EM_inferred_label_confidence_{who}'] = np.nan
            consensuses[f'EM_inferred_label_count_{who}'] = np.nan
            consensuses[f'UNBIASED_EM_inferred_label_{who}'] = np.nan
            consensuses[f'UNBIASED_EM_inferred_label_confidence_{who}'] = np.nan  # noqa
            # number of Ps in THIS SET who agree with the unbiased label
            consensuses[f'UNBIASED_EM_inferred_label_count_{who}'] = np.nan
            # Soft EM probabilities for various classes (using THIS evalset)
            consensuses.update({
                f'EM_prob_{cls}_{who}': np.nan
                for cls in ['undetected'] + ir.CLASSES
            })

        anch.update(consensuses)
        # per-user and eval-set labels
        anch.update({f'{usr}': usrs[evalset][usr] for usr in ir.All})
        # for convenience, add manual boundary coordinates
        if manual_boundary is not None:
            anch.update({f'manual_{k}': v for k, v in manual_boundary.items()})
        else:
            anch.update({f'manual_coords_{c}': np.nan for c in ('x', 'y')})
        # for convenience, add algorithmic boundary coordinates
        anch.update(_get_algorithmic_boundary_for_evalset(evalset))

        # add information about algorithmic boundary DICE stats
        if anch['has_manual_boundary'] and anch['has_algorithmic_boundary']:

            # get coords list
            manual_coords = _get_coords_from_coordstr(
                anch['manual_coords_x'], anch['manual_coords_y'])
            algorithmic_coords = _get_coords_from_coordstr(
                anch['algorithmic_coords_x'], anch['algorithmic_coords_y'])

            # make sure sizes match
            all_coords = np.array(manual_coords + algorithmic_coords)
            bound = {}
            bound['min_x'], bound['min_y'] = np.min(all_coords, axis=0)
            bound['max_x'], bound['max_y'] = np.max(all_coords, axis=0)

            # create masks
            manual = create_mask_from_coords(np.int32(manual_coords), **bound)
            algorithmic = create_mask_from_coords(
                np.int32(algorithmic_coords), **bound)

            # now get intersection, union, etc
            intersect = np.sum((manual + algorithmic) == 2)
            sums = np.sum(manual) + np.sum(algorithmic)
            anch['algorithmic_vs_manual_intersect'] = intersect
            anch['algorithmic_vs_manual_sums'] = sums
            anch['algorithmic_vs_manual_IOU'] = intersect / (sums - intersect)
            anch['algorithmic_vs_manual_DICE'] = 2. * intersect / sums

        # for completeness, and to be able to access these later, we also
        # save a comma-separated list of the girder IDs of annotations
        # that matched to this medoid from this evaluation set
        keep = relevant_contours.loc[:, 'evalset'] == evalset
        anch['matches'] = ",".join(list(relevant_contours.loc[keep, :].index))
        return anch

    # Get user label tally, as well as for each experience level. This also
    # updates the usr dict as an intentional side effect
    anninfo = {
        ev: _get_anninfo_for_subgroup(ev) for ev in ir.EVALSET_NAMES}

    # Get the anchor bounding box
    anchorbbox, manual_boundary = _get_anchor_bbox(anninfo)

    # Initialize the anchor, making sure all fields are there
    anchor = {
        evalset: _get_anchor_dict(evalset)
        for evalset in ir.EVALSET_NAMES}

    return anchor


def _get_anchors_for_fov_by_clustering(
        all_conts, iou, participants, min_iou=0.5, constrained=True):
    """Get nucleus anchors for FOV.

    Parameters
    ----------
    all_conts: DataFrame
    iou: DataFrame
        IOU for annotations against each other
    participants: dict
        each key is an eval set, and its value is a list of
        participants who annotated this FOV
    min_iou: float
        min_iou for clustering (1 - linkage threshold)
    constrained: bool
        constrained clustering? i.e. prevent annotations by the same
        participant from appearing in the same cluster?

    Returns
    -------
    DataFrame

    """
    # first we cluster
    model = _get_clusters_for_fov(
        all_conts=all_conts, iou=iou, min_iou=min_iou, constrained=constrained)

    # Differentiate those who didn't annotated FOV and those who did
    # this is an INITIALIZATION dict so a true copy must be passed
    usrs = {
        evalset: {
            usr: 'undetected' if usr in ppl else 'DidNotAnnotateFOV'
            for usr in ir.All
        } for evalset, ppl in participants.items()
    }

    cluster_anchors = {evalset: [] for evalset in ir.EVALSET_NAMES}

    # clid=4427; annlocs=model.clusters[clid]
    for clid, annlocs in model.clusters.items():
        relevant_idxs = list(iou.index[annlocs])
        anchor = _get_anchor_from_single_cluster(
            relevant_contours=all_conts.loc[relevant_idxs, :].copy(),
            cluster_iou=iou.iloc[annlocs, annlocs].copy(),
            usrs=copy.deepcopy(usrs),
        )
        for evalset in anchor.keys():
            cluster_anchors[evalset].append(anchor[evalset])

    # convert to dfs
    for evalset in ir.EVALSET_NAMES:
        cluster_anchors[evalset] = DataFrame.from_records(
            cluster_anchors[evalset])
        cluster_anchors[evalset].index = cluster_anchors[
            evalset].loc[:, 'anchor_id']
        cluster_anchors[evalset].loc[:, "min_iou"] = min_iou

    return cluster_anchors


def _get_bounds_for_eval_fov(gc, dbcon, elementid, MPP, MAG, all_conts):
    # any copy of the slide will do obviously
    slide_id = read_sql_query(f"""
        SELECT "itemId"
        FROM "annotation_docs"
        WHERE "annotation_girder_id" IN (
            SELECT "annotation_girder_id"
            FROM "annotation_elements"
            WHERE "element_girder_id" = "{elementid}"
        ) 
        ;""", dbcon).iloc[0, 0]

    # calculate the scale factor
    sf, appendStr = get_scale_factor_and_appendStr(
        gc=gc, slide_id=slide_id, MPP=MPP, MAG=MAG)

    # get overall bounds for FOV
    bounds = {
        'XMIN': np.min(all_conts.loc[:, "xmin"]),
        'YMIN': np.min(all_conts.loc[:, "ymin"]),
        'XMAX': np.max(all_conts.loc[:, "xmax"]),
        'YMAX': np.max(all_conts.loc[:, "ymax"]),
        'sf': sf,
        'appendStr': appendStr,
    }

    return bounds, slide_id


def _get_rgb_for_interrater(gc, bounds, slide_id):
    """"""
    getStr = \
        "/item/%s/tiles/region?left=%d&right=%d&top=%d&bottom=%d" \
        % (slide_id,
           bounds['XMIN'], bounds['XMAX'],
           bounds['YMIN'], bounds['YMAX'])
    getStr += bounds['appendStr']
    resp = gc.get(getStr, jsonResp=False)
    rgb = get_image_from_htk_response(resp)

    return rgb


def _visualize_bboxes_on_rgb(
        rgbim, xy_df, fovcont=None,
        totalcount=None, lblcount=None, lblcolors=None,
        bbox_linewidth=None, bbf=0.1, bbox_color='#525150',
        add_points=False, point_size=None, psmin=4, psf=0.3,
        point_color='#525150'):

    fov = copy.deepcopy(fovcont)

    # later on flipped by matplotlib for weird reason
    rgb = np.flipud(rgbim.copy())

    # make sure also y coords are flipped
    for locstr in ("ymin", "ymax"):
        xy_df.loc[:, locstr] = rgb.shape[0] - xy_df.loc[:, locstr] + 1
        if fovcont is not None:
            fov[locstr] = rgb.shape[0] - fovcont[locstr] + 1

    if add_points:
        fig = plt.figure()
        dpi = 300
    else:
        fig = plt.figure(
            figsize=(rgb.shape[1] / 1000, rgb.shape[0] / 1000), dpi=100)
        dpi = 1000

    ax = plt.subplot(111)
    ax.imshow(rgb)

    plt.axis('off')
    ax = plt.gca()
    ax.set_xlim(0.0, rgb.shape[1])
    ax.set_ylim(0.0, rgb.shape[0])

    # single versus multiple bounding box linewidths
    if np.isscalar(bbox_linewidth):
        lw = [bbox_linewidth] * xy_df.shape[0]
    else:
        lw = bbf * totalcount

    # single versus multiple bounding box colors
    if np.isscalar(bbox_color):
        ec = [bbox_color] * xy_df.shape[0]
    else:
        ec = lblcolors

    # single versus multiple point sizes
    if add_points:
        if np.isscalar(point_size):
            ps1 = [point_size] * xy_df.shape[0]
            ps2 = None
        else:
            ps1 = psmin + totalcount * psf
            ps2 = psmin + lblcount * psf

    # now draw the bounding boxes & add points
    loc = 0
    for _, me in xy_df.iterrows():

        # add bounding box (detection)
        rect = patches.Rectangle(
            xy=(me['xmin'], me['ymin']),
            width=me['xmax'] - me['xmin'],
            height=me['ymax'] - me['ymin'],
            edgecolor=ec[loc], linewidth=lw[loc],
            facecolor='none', linestyle='-')
        ax.add_patch(rect)
        loc += 1

    # Add the fov contour if given
    if fovcont is not None:
        rect = patches.Rectangle(
            xy=(fov['xmin'], fov['ymin']),
            width=fov['xmax'] - fov['xmin'],
            height=fov['ymax'] - fov['ymin'],
            edgecolor=fov['color'],
            linewidth=0.2 if add_points else 0.2,
            facecolor='none', linestyle='--')
        ax.add_patch(rect)

    # add point (classification) -- looks ugly (yikes!)
    if add_points:
        x = np.int32(xy_df.loc[:, "xmin"] + (
                xy_df.loc[:, "xmax"] - xy_df.loc[:, "xmin"]) / 2)
        y = np.int32(xy_df.loc[:, "ymin"] + (
                xy_df.loc[:, "ymax"] - xy_df.loc[:, "ymin"]) / 2)
        # main seed points (totals)
        ax.scatter(
            x, y, color=point_color, alpha=1., marker='.',
            edgecolor=None, s=ps1 ** 2,
        )
        # most dominant label
        if ps2 is not None:
            ax.scatter(
                x, y, color=lblcolors, alpha=1.0, marker='.',
                edgecolor=None, s=ps2 ** 2,
            )

    ax.axis('off')
    fig.subplots_adjust(bottom=0, top=1, left=0, right=1)

    buf = io.BytesIO()
    plt.savefig(buf, format='png', pad_inches=0, dpi=dpi)
    buf.seek(0)
    rgb_vis = np.uint8(Image.open(buf))[..., :3]
    plt.close()

    return rgb_vis


def _restrict_to_user_subset(all_conts, who='NP'):
    keep = all_conts.loc[:, "user"].isin(ir.who[who])
    return all_conts.loc[keep, :]


def _get_annlocs_for_same_user(all_conts):
    """Find annotation indices that cannot appear in the same cluster.

    In this case, annotations from the same user and evaluation set
    CANNOT map to the same anchor since the user's intention, by definition,
    is to annotate two separate nuclei.
    """
    dontlink = []
    for evalset in ir.EVALSET_NAMES:
        evalconts = all_conts.loc[all_conts.loc[:, "evalset"] == evalset, :]
        for user in ir.All:
            user_elements = evalconts.loc[evalconts.loc[:, "user"] == user, :]
            if user_elements.shape[0] > 0:
                dontlink.append(np.argwhere(np.in1d(
                    all_conts.index, user_elements.index))[:, 0])
    return dontlink


def get_anchors_for_iou_thresh(
        fovinfo, dbcon, who='All', min_iou=0.25, constrained=True, gc=None,
        add_relative_bounds=True, MPP=0.2, MAG=None):
    """Get all nucleus anchors for a range of clustering IOU thresholds.

    Note that there are two types of analyses we want to investigate:
    1. Can we use concordance with SPs as proxy for core set performance
    2. What would happen if we only use NP annotations for everything
    See ctme/nuclei/ideas for details ...

    Q1: Since the CORE_SET is based on the same methodology as EVAL_SET_3,
      how accurate are NP annotations compared to SPs/JPs on EVAL_SET_3. i.e.
      if SPs/JPs did the annotations under the exact same conditions?
    Q2: How accurate are the annotations made by SPs & NPs, in absolute terms,
      when compared against our "unbiased" approach (SPs/JPs on EVAL_SET_1)??

    Parameters
    ----------
    fovinfo: dict
        keys are names of participants who annotated the fov, each entry
        is also a dict constaining metadata about the fov by this particular
        participant
    dbcon: connected sqlalchemy database
    who: str
        participant experience level to keep
    min_iou: float
        min_iou to map annotations to same cluster
    constrained: bool
    gc: authenticated girder client
    add_relative_bounds: bool
    MPP: float or None
    MAG: float or None

    Returns
    -------
    dict

    """
    out = dict()

    # Read all contours. Since the masks are small and there's < 30 nuclei
    # this is OK and it save us IO cost
    all_conts, out['fov_conts'] = _get_all_contours_for_eval_fov(
        fovinfo=fovinfo, dbcon=dbcon)

    # we get the absolute bounds of the FOV, including all annotations
    # from all users and evaluation sets
    if add_relative_bounds:
        assert gc is not None, "You must provide girder client"
        out['bounds'], out['slide_id'] = _get_bounds_for_eval_fov(
            gc=gc, dbcon=dbcon, elementid=all_conts.index[0],
            MPP=MPP, MAG=MAG, all_conts=all_conts)

    # ** Restrict to subsets of users **
    if who != 'All':
        all_conts = _restrict_to_user_subset(all_conts, who=who)

    # If empty FOV, return None
    if all_conts.shape[0] < 1:
        return

    # differentiate between bboxes and polygons
    all_conts.loc[:, 'type'] = all_conts.loc[:, 'coords_x'].apply(
        lambda x: 'polyline' if len(x.split(',')) > 5 else 'rectangle')

    # get ious for all potential nuclei bboxes
    iou = _get_iou_for_fov(all_conts)

    # init
    all_anchors = {evset: DataFrame() for evset in ir.EVALSET_NAMES}

    # This is a list of people who annotated the FOV for both the main
    # evaluation set as well as the controls
    participants = {
        k: list(fovinfo[k].keys()) if k in fovinfo.keys() else []
        for k in ir.EVALSET_NAMES
    }

    # get absolute coordinates of medoids
    cluster_anchors = _get_anchors_for_fov_by_clustering(
        all_conts=all_conts, iou=iou, min_iou=min_iou,
        constrained=constrained, participants=participants)

    for evset in ir.EVALSET_NAMES:
        # relative to the fov at desired MPP
        if add_relative_bounds:
            relative_anchors = _get_relative_anchors(
                cluster_medoids=cluster_anchors[evset].loc[
                    :, ["xmin", "ymin", "xmax", "ymax"]],
                bounds=out['bounds'])
            for col in relative_anchors.columns:
                cluster_anchors[evset].loc[:, f'{col}_relative'] = \
                    relative_anchors.loc[:, col]

        all_anchors[evset] = concat(
            (all_anchors[evset], cluster_anchors[evset]), axis=0,
            sort=False, ignore_index=True)

    out['all_anchors'] = all_anchors

    return out


def _get_fovnames_with_commonest_nobservers(
        fovmetas: DataFrame, who: str) -> Tuple[List[str], int]:
    """"Get the names of FOVs with commonest number of observers.

    Most of the time, this will be all observers (i.e. 6 of 6 pathologists),
    but there may be exceptions. It is important to only keep FOVs with the
    same number of observers so that the ease of detection of nuclei is
    not confounded by the number who actually annotated the FOV. i.e so that
    we know that a nucleus detected by only 4 pathologists is because it is
    a tough nucleus, not because only 4 pathologists happened to annotate
    this particular FOV.
    """
    nperfov = {
        row['fovname']: len([
            p for p in row['participants'].split(',')
            if p in ir.who[who]
        ])
        for _, row in fovmetas.iterrows()
    }
    tally = Counter([v for _, v in nperfov.items()])
    # the higher of the two most common n_observers per FOV
    maxn = max(tally.most_common()[0][0], tally.most_common()[1][0])
    fovnames = [k for k, v in nperfov.items() if v == maxn]
    return fovnames, maxn


def get_fovs_annotated_by_almost_everyone(
        dbcon_anchors, unbiased_is_truth: bool, whoistruth: str,
        evalset: str, who: str, get_anchors: bool = True) -> Dict[str, Any]:
    """Get FOVs with most observers.

    See docstring for _get_fovnames_with_commonest_nobservers().
    """
    # get fov metadata
    fovmetas = read_sql_query(f"""
        SELECT  "fovname", "participants_{evalset}" AS "participants"
        FROM "fov_meta"
        WHERE "participants_{evalset}" NOT NULL
    ;""", dbcon_anchors)

    # Get fovs with commonest no. of observers at this experience level
    fovnames, maxn = _get_fovnames_with_commonest_nobservers(
        fovmetas=fovmetas, who=who)

    if maxn < 1:
        return

    # get anchors and matches for relevant FOVs
    # IMPORTANT!!! WE'RE ONLY SHOWING STATS FOR THESE FOVS SO THAT
    # THE NUCLEUS DETECTION AGREEMENT STATS MAKE SENSE AND ARE CONTROLLED\
    # FOR THE NUMBER OF PEOPLE WHO ACTUALLY ANNOTATED THE FOVS
    if get_anchors:
        fovidstr = ",".join([f'"{j}"' for j in fovnames])
        ubstr = ir._ubstr(unbiased_is_truth)
        tablename = f'v3.1_final_anchors_' \
                    f'{evalset}_{ubstr}{whoistruth}_AreTruth'
        anchors = read_sql_query(f"""
            SELECT  * FROM "{tablename}"
            WHERE "fovname" IN ({fovidstr})
        ;""", dbcon_anchors)
    else:
        anchors = None

    return {
        'fovmetas': fovmetas,
        'fovnames': fovnames,
        'maxn': maxn,
        'anchors': anchors,
    }


def _get_custom_uniform_cmap(r, g, b, cmax=256, N=256, more_is_dark=True):
    vals = np.ones((N, 4))
    vals[:, 0] = np.linspace(r / cmax, 1, N)
    vals[:, 1] = np.linspace(g / cmax, 1, N)
    vals[:, 2] = np.linspace(b / cmax, 1, N)
    if more_is_dark:
        vals = np.flipud(vals)
    return ListedColormap(vals)


def _connect_to_sqlite(db_path: str):
    sql_engine = create_engine('sqlite:///' + db_path, echo=False)
    return sql_engine.connect()


def _connect_to_anchor_db(savepath: str, constrained: bool = True):
    constr = '' if constrained else '_unconstrained'
    db_path = opj(savepath, 'i1_anchors', f'anchors{constr}' + '.sqlite')
    return _connect_to_sqlite(db_path)


def _annotate_krippendorph_ranges(axis, minx, maxx, shades=False):
    """Add the agreement ranges for Krippendorph alpha subplot."""
    if shades:
        # fill agreement ranges
        kdict = ir.KAPPA_RANGES[2]
        axis.fill_between(
            x=[minx, maxx],
            y1=kdict['min'], y2=kdict['max'],
            color='green', alpha=kdict['max'] * 0.2,
            label=f'{kdict["descr"]} Agreement')
    else:
        # use lines to annotate ranges
        for kno, kdict in enumerate(ir.KAPPA_RANGES[::-1]):
            lgnd = {'label': 'Agreement level'} if kno == 0 else {}
            # if kdict['descr'] != 'Slight':
            axis.axhline(
                kdict['min'], linestyle='dotted', c='gray', alpha=0.7,
                **lgnd
            )
            axis.annotate(
                kdict['descr'], xy=(maxx - 0.42, kdict['min'] + 0.02),
                ha='center', va='bottom', color='gray', alpha=0.7)


def calc_stats_simple(*args, **kwargs):
    """Calculate simple stistics"""
    return calculate_4x4_statistics(*args, **kwargs)


def get_roc_and_auroc_for_who(
        dbcon=None, evalset=None, anchors=None,
        truthcol=None, probcol_prefix='', probcol_postfix='',
        who='NPs', whoistruth='Ps', unbiased_is_truth=False, clsgroup=None):
    """Get the AUC and ROC accuracy for inferred labels from NPs compared to
    the "real" inferred truth from Ps.
    """
    if anchors is None:
        assert evalset is not None
        ubstr = ir._ubstr(unbiased_is_truth)
        # Get "true" inferred labels (from SPs) and EM inferred label
        # probabilities (from NPs)
        anchors = ir._get_true_and_inferred_labels_for_who(
            dbcon=dbcon, whoistruth=whoistruth, unbiased=unbiased_is_truth,
            who=who, evalset=evalset, colnames=['*'])

        # maybe remap classes
        if clsgroup is not None:
            anchors = remap_classes_in_anchorsdf(
                anchors=anchors, clsgroup=clsgroup)

        truthcol = f'{ubstr}EM_inferred_label_{whoistruth}'
        probcol_prefix = 'EM_prob_'
        probcol_postfix = f'_{who}'
        ilabelcol = f'EM_inferred_label_{who}'
    else:
        assert truthcol is not None
        ilabelcol = 'ilabel'

    # get rid of "undetected"
    anchors = anchors.loc[anchors.loc[:, ilabelcol] != 'undetected', :]

    # The following implementation is based on sklarn's
    # documentation at:
    # https://scikit-learn.org/stable/auto_examples/
    # model_selection/plot_roc.html#sphx-glr-auto-examples-
    # model-selection-plot-roc-py

    # map class names to code (for roc-auc) .. note that NOT all
    # classes are represented .. only those for which at least one
    # example is there in the truth, otherwise AUC is not defined
    y_true = anchors.loc[:, truthcol]
    rclasses = np.unique(y_true).tolist()
    rclassmap = {j: i for i, j in enumerate(rclasses)}
    n_rclasses = len(rclasses)
    int_rclasses = list(range(n_rclasses))

    # Get one-hot truth
    y_true = y_true.map(rclassmap).values
    y_true = label_binarize(y_true, classes=int_rclasses)

    # Renormalize to 1 (since we're not including "undetected")
    y_scores = anchors.loc[:, [
        f'{probcol_prefix}{cls}{probcol_postfix}' for cls in rclasses]].values
    keep = np.sum(y_scores, axis=1) > 0  # avoid division by zero
    y_true = y_true[keep, :]
    y_scores = y_scores[keep, :]
    y_scores = y_scores / np.sum(y_scores, axis=1).reshape(
        y_scores.shape[0], 1)

    # Compute ROC curve and ROC area for each class (one-versus-rest, "ovr")
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i, cls in enumerate(rclasses):
        fpr[cls], tpr[cls], _ = roc_curve(
            y_true[:, i], y_scores[:, i])
        roc_auc[cls] = auc(fpr[cls], tpr[cls])

    # Compute micro-average ROC curve and ROC area
    fpr['micro'], tpr['micro'], _ = roc_curve(
        y_true.ravel(), y_scores.ravel())
    roc_auc['micro'] = auc(fpr['micro'], tpr['micro'])

    # Now calculate the macro average

    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[cls] for cls in rclasses]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for cls in rclasses:
        mean_tpr += np.interp(all_fpr, fpr[cls], tpr[cls])

    # Finally average it and compute AUC
    mean_tpr /= n_rclasses

    fpr['macro'] = all_fpr
    tpr['macro'] = mean_tpr
    roc_auc['macro'] = auc(fpr['macro'], tpr['macro'])

    return tpr, fpr, roc_auc


def get_precision_recall_for_who(
        dbcon=None, evalset=None, who='NPs', whoistruth='Ps',
        unbiased_is_truth=False,):
    """Get the precision and recall for inferred labels from NPs compared to
    the "real" inferred truth from Ps.
    """
    ubstr = ir._ubstr(unbiased_is_truth)
    tablename = f'all_anchors_{evalset}'
    truthcol = f'{ubstr}EM_inferred_label_{whoistruth}'
    probcol_prefix = 'EM_prob_'
    probcol_postfix = f'_{who}'
    colnames = ['anchor_id', f'{truthcol}', f'EM_prob_undetected_{who}']
    anchors = read_sql_query(f"""
        SELECT {ir._get_sqlitestr_for_list(colnames)}
        FROM "{tablename}"
        WHERE "n_matches_{who}" >= 2 
          AND "EM_inferred_label_{who}" NOT NULL
    ;""", dbcon)

    y_true = anchors.loc[:, truthcol].copy()
    y_true[y_true.isnull()] = 'undetected'
    y_true = 1 - (y_true == 'undetected').values

    scorecol = f'{probcol_prefix}undetected{probcol_postfix}'
    y_score = 1 - anchors.loc[:, scorecol].values

    # a random classifier simply predicts the percent true class
    # i.e. is a horizontal line at the percentage of labels that are "true"
    out = {'random': np.mean(y_true)}

    out['precision'], out['recall'], out['thresholds'] = \
        precision_recall_curve(y_true=y_true, probas_pred=y_score)
    out['AP'] = average_precision_score(y_true, y_score)

    return out


def _get_clmap(clsgroup):
    assert clsgroup in ['raw', 'main', 'super']
    ncg = NucleusCategories
    if clsgroup == 'main':
        clmap = deepcopy(ncg.raw_to_main_categmap)
        class_list = deepcopy(ncg.main_categs)
    elif clsgroup == 'super':
        clmap = deepcopy(ncg.raw_to_super_categmap)
        class_list = deepcopy(ncg.super_categs)
    else:
        clmap = None
        class_list = deepcopy(ncg.raw_categs)
    return clmap, class_list


def _update_majority_vote(dfrow, class_list):
    who_labels = {}
    for who in ['NP', 'P']:
        who_labels[who] = {}
        ss = ['JP.', 'SP.'] if who == 'P' else [f'{who}.']
        wlabs = {
            k: v for k, v in dfrow.items()
            if any([k.startswith(s) for s in ss])}
        wlabs = {k: v for k, v in wlabs.items() if v in class_list}
        if len(wlabs) > 0:
            cnts = Counter(wlabs.values()).most_common()[0]
            dfrow[f'MV_inferred_label_{who}s'] = cnts[0]
            dfrow[f'MV_inferred_label_count_{who}s'] = cnts[1]
            who_labels[who].update(wlabs)

    return dfrow, who_labels


def _update_em_inferred_label(dfrow, clmap, class_list, who_labels):
    clist = class_list + ['undetected']
    for who in ['NP', 'P']:

        # aggregate probabilities from children (eg. vasc.end & fibro)
        maxpr = 0.
        ilab = {}
        for cl in clist:
            children = [
                k for k, v in clmap.items() if
                (v == cl) and not k.startswith('correction')]
            pr = np.nansum([dfrow[f'EM_prob_{ch}_{who}s'] for ch in children])
            dfrow[f'EM_prob_{cl}_{who}s'] = pr
            if np.isfinite(pr) and (pr > maxpr):
                ilab.update({'cls': cl, 'pr': pr})
                maxpr = pr

        # add inferred label
        if len(ilab) > 0:
            dfrow[f'EM_inferred_label_{who}s'] = ilab['cls']
            dfrow[f'EM_inferred_label_confidence_{who}s'] = ilab['pr']
            dfrow[f'EM_inferred_label_count_{who}s'] = len(
                [j for j in who_labels[who].values() if j == ilab['cls']])

    return dfrow


def remap_classes_in_anchorsdf(
        anchors, clsgroup, also_ilabel=True, remove_ambiguous=True,
        who_determines_ambig='Ps', how_ambig_is_determined='EM'):
    """Take an anchors table slice & remap classes, & remap truth by
    aggregating EM probabs to be used for Krippendorph, accuracy, etc
    for example: from raw annotations -> main classes
    """
    # TODO: currently, only mapping within the SAME evalset is supported
    relevant = [j for j in anchors.columns if not j.startswith('UNBIASED_')]
    anchors = anchors.loc[:, relevant]

    # edge case
    if anchors.shape[0] < 1:
        return anchors

    # reset inferred labels
    colns = [
        j for j in anchors.columns for s in ['MV', 'EM']
        if j.startswith(f'{s}_inferred_label')]
    anchors.loc[:, colns] = np.nan

    # get class mapping
    clmap, class_list = _get_clmap(clsgroup)
    for extra in [
            'non-existent', 'undetected', 'DidNotAnnotateFOV',
            np.nan, None]:
        clmap[extra] = extra

    # simple remap classes (i.e. just replace label by parent)
    simple_maps = [
        j for j in anchors.columns for start in ['NP.', 'JP.', 'SP.']
        if j.startswith(start)
    ]
    slc = anchors.loc[:, simple_maps]
    anchors.loc[:, simple_maps] = slc.applymap(lambda x: clmap[x])

    # update inferred labels
    def _update_row(row):
        row = dict(row)
        row, who_labs = _update_majority_vote(
            dfrow=row, class_list=class_list)
        row = _update_em_inferred_label(
            dfrow=row, clmap=clmap, class_list=class_list,
            who_labels=who_labs)
        return Series(row)

    if also_ilabel:
        anchors = anchors.apply(lambda r: _update_row(r), axis=1)

        # remove anchors that determined to be "unreal" (ambiguous)
        # by, say, pathologists
        if remove_ambiguous:
            wda = who_determines_ambig
            had = how_ambig_is_determined
            truthcol = f'{had}_inferred_label_{wda}'
            anchors = anchors.loc[
                anchors.loc[:, f'{truthcol}'] != 'AMBIGUOUS', :]

    return anchors
