import numpy as np
from pandas import DataFrame as df, concat
import random
import string
from skimage.measure import regionprops

from algorithmic_suggestions.data_management import get_fov_bounds
from algorithmic_suggestions.SQLite_Methods import SQLite_Methods


def preprocess_mrcnn_output_tile(tile, instance_bias):
    """
    Preprocess mrcnn output tile
    """
    
    def add_instance_bias(mask, instance_bias):
        if instance_bias > 0:
            mask_original = mask.copy()
            mask = mask + instance_bias
            mask[mask_original == 0] = 0
        return mask
    
    # exclude pixels in solid instances without corresponding contour
    # notice we're using the invert= True option
    exclude = np.isin(tile[..., 1], np.unique(tile[..., -1]), invert= True)
    tile[exclude] = 0
        
    # make sure nuclear instances are contigous
    # this is important to make sure instances from different
    # tiles don't get the same label
    unique_nuclei = list(np.unique(tile[..., 1]))
    unique_nuclei.remove(0)
    contiguous = np.zeros(tile[..., 1].shape)
    for instanceidx, instance in enumerate(unique_nuclei):
        contiguous[tile[..., 1] == instance] = instanceidx+1
    tile[..., 1] = contiguous.copy()
    
    # instance bias to make sure instance labels are separate
    tile[..., 1] = add_instance_bias(tile[..., 1].copy(), instance_bias)
    tile[..., -1] = add_instance_bias(tile[..., -1].copy(), instance_bias)
    
    return tile


def reconcile_mrcnn_with_regions(
        mrcnn, regions, codes_region, code_rmap_dict, 
        KEEP_CLASSES_CODES):
    """
    Reconciles mrcnn predictions with region labels.
    Args:
        mrcnn - [m,n,4] mrcnn output
        regions - [m,n]
        codes_region - dictionary mapping class to region GT code
        code_rmap_dict -  dict mapping mrcnn codes to region codes
        KEEP_CLASSES_CODES - list of region codes for classes to keep
           in mrcnn output without reconciliation with region codes
    Returns:
        mrcnn postprocessed
    """
    # re-map mrcnn labels to standardized codes (region)
    label_channel = mrcnn[..., 0].copy()
    for k in np.unique(label_channel):
        label_channel[label_channel == k] = code_rmap_dict[k]
    mrcnn[..., 0] = label_channel.copy()
    del label_channel
    
    # Ignore areas outside ROI 
    outside = np.argwhere(regions == codes_region["background"])
    mrcnn[outside[:,0], outside[:,1], :] = codes_region["background"]
    
    # Nuclei touching rare regions get assigned region class
    unique_regions = np.unique(regions)
    unique_regions_rare = [j for j in unique_regions if j not in KEEP_CLASSES_CODES]
    
    for k in unique_regions_rare:
        
        # first we get pixel values associated with unique 
        # touching nuclei. NOTE: we cannot just naively assign 
        # rare class pixels the same label in mrcnn output
        # as that would not correspond to detected nuclear
        # boundaries by mrcnn (i.e. would "chop off") nuclei
        # at edge of rare region
        touching_nuclei = mrcnn[..., 1].copy()
        touching_nuclei[regions != k] = 0
        touching_nuclei_instances = [int(j) for j in list(np.unique(touching_nuclei))]
        touching_nuclei_instances.remove(0)
        
        # now assign to mask    
        rare_nuclei = np.isin(mrcnn[..., 1], touching_nuclei_instances)
        label_channel = mrcnn[..., 0]
        label_channel[rare_nuclei] = k
        mrcnn[..., 0] = label_channel.copy()
    
    return mrcnn


def get_nuclei_props_and_outliers(
    imname, mrcnn, props_fraction, ignore_codes=[0,]):
    """
    Gets nuleus AnnotsDF, including nuclei props and gets
    unique nucleus ids of nuclei that have weird shape or are probably 
    segmentation artifacts (in comparison to others of the same label) 
    in terms of area or shape properties.
    Args:
        * mrcnn [m, n, 4] - postprocessed outpt from mrcnn (4 channels)
        * props_fraction - fraction of top and bottom x percentil 
                        ot nuclei to flag for review by area/shape
        * ignore_codes - list of region codes for nuclei classes to ignore. 
    Returns:
        AnnotsDF - 
        extreme_nuclei - 
        segmentation artifacts -
    """
    # Prep to save nucleus info
    # ==========================================================

    # basic info that would be assigned to all nuclei
    slide_name = imname.split("_")[0]
    roi_xmin = int(imname.split("_xmin")[1].split("_")[0])
    roi_ymin = int(imname.split("_ymin")[1].split("_")[0])
    roi_offset_xmin_ymin = "%d,%d" % (roi_xmin, roi_ymin)

    # Init df to save coords for this image
    Annots_DF = df(columns= [
        "unique_nucleus_id",
        "slide_name", 
        "nucleus_label", 

        "area",
        "extent",
        "aspect",
        "circularity",    
        "nucleus_label_confidence",

        "roi_offset_xmin_ymin", 
        "center_relative_to_slide_x_y",
        "bounding_box_relative_to_slide_xmin_ymin_xmax_ymax",
        "boundary_relative_to_slide_x_coords",
        "boundary_relative_to_slide_y_coords",
    ])

    def add_info_to_Annots_DF(prop, lbl, aspect, circularity):
        """
        Adds prop information to annotation dataframe
        """
        center_y, center_x = prop.centroid
        center_x = int(center_x) + roi_xmin
        center_y = int(center_y) + roi_ymin

        # Add identification info, including unique ID
        gibbrish = ''.join(random.choice(
            string.ascii_uppercase + string.digits) for _ in range(5))
        unique_nucleus_id = "%s_%d_%d_%s" % (slide_name, center_x, center_y, gibbrish)

        Annots_DF.loc[instance_id, "unique_nucleus_id"] = unique_nucleus_id
        Annots_DF.loc[instance_id, "slide_name"] = slide_name
        Annots_DF.loc[instance_id, "roi_offset_xmin_ymin"] = roi_offset_xmin_ymin

        Annots_DF.loc[instance_id, "center_relative_to_slide_x_y"] = "%d,%d" % (center_x, center_y)
        Annots_DF.loc[instance_id, "nucleus_label"] = lbl

        # add area/shape props
        Annots_DF.loc[instance_id, "area"] = int(prop.area)
        Annots_DF.loc[instance_id, "extent"] = int(100 * prop.extent)
        Annots_DF.loc[instance_id, "aspect"] = int(100 * aspect)
        Annots_DF.loc[instance_id, "circularity"] = int(100 * circularity)
        Annots_DF.loc[instance_id, "nucleus_label_confidence"] = int(100 * mrcnn[first_coord_row, first_coord_col, 2])

        # Add coordinates
        coords_x = prop.coords[:, 1] + roi_xmin
        coords_y = prop.coords[:, 0] + roi_ymin

        Annots_DF.loc[instance_id, "bounding_box_relative_to_slide_xmin_ymin_xmax_ymax"] = \
            "%d,%d,%d,%d" % (prop.bbox[1] + roi_xmin, prop.bbox[0] + roi_ymin, 
                             prop.bbox[3] + roi_xmin, prop.bbox[2] + roi_ymin)
        Annots_DF.loc[instance_id, "boundary_relative_to_slide_x_coords"] = \
            ",".join([str(j) for j in coords_x])
        Annots_DF.loc[instance_id, "boundary_relative_to_slide_y_coords"] = \
            ",".join([str(j) for j in coords_y])

    # Go through the unique labels and add info
    # ==========================================================
    
    unique_labels = [int(j) for j in np.unique(mrcnn[..., 0]) if j != 0]
    extreme_nuclei = []
    segmentation_artifacts = []
    
    # lblidx= 0; lbl= unique_labels[lblidx]
    for lblidx, lbl in enumerate(unique_labels):

        print("\tGetting props for", lbl)
        this_lbl = 0 + (mrcnn[..., 0] == lbl)
        nuclei = mrcnn[..., 1].copy()
        nuclei[this_lbl == 0] = 0
        props = regionprops(label_image= np.int32(nuclei), coordinates='rc')

        if len(props) < 3:
            continue

        # get number of nuclei in top or bottom x percentile
        n_props = len(props)
        n_props_review = int((props_fraction * n_props) / 2) # top and bottom

        # init props df for this label
        # Note that we do things that way because how "extreme" each
        # nucleus is is judges in comparison to other nuclei with the
        # same label (i.e. tumor compared to tumor, lymphocyte to lymphocyte etc)
        propnames = ["instance_id", "area", 
                     "extent", "circularity", "aspect"]
        relevant_props = df(index= np.arange(n_props), columns= propnames)

        # Go through each nucleus prop
        # ==========================================================

        # propidx= 0; prop = props[propidx]
        for propidx, prop in enumerate(props):

            if propidx % 500 == 0:
                print("\t\tprop %d of %d" % (propidx, n_props))

            # Calculate stats we care about and assign to prop df
            # ====================================================

            first_coord_row  = int(prop.coords[0,0])
            first_coord_col = int(prop.coords[0,1])
            instance_id = int(mrcnn[first_coord_row, first_coord_col, 1])

            if instance_id == 0:
                continue 
            relevant_props.loc[propidx, "instance_id"] = instance_id
            relevant_props.loc[propidx, "area"] = prop.area
            relevant_props.loc[propidx, "extent"] = prop.extent 
            aspect = 0
            circularity = 0
            try:
                if np.isfinite(prop.major_axis_length) and (
                    prop.major_axis_length > 0):
                    aspect = prop.minor_axis_length / prop.major_axis_length
                    relevant_props.loc[propidx, "aspect"] = aspect
            except ValueError:
                pass
            try:
                if np.isfinite(prop.perimeter) and (
                    prop.perimeter > 0):
                    circularity = (4 * np.pi * prop.area) / (prop.perimeter ** 2)
                    relevant_props.loc[propidx, "circularity"] = circularity
            except ValueError:
                pass

            # Assign to AnnotDF
            # ====================================================
            add_info_to_Annots_DF(prop= prop, lbl= lbl, 
                                  aspect=aspect, circularity=circularity)

        # Get outliers for this label (eg weird tumor nuclei)
        # ==========================================================
        
        # don't add outliers for exclude and other useless classes
        if lbl in ignore_codes:
            continue

        # for each property, get candidate nuclei that are extreme
        # we give special attention to large blobs for review in
        # so we don't count them as "extreme" but as a separate
        # category of flagged nuclei for review (since these are
        # most probably segmentation artifacts from clustered nuclei)
        candidates = []
        artifact_candidates = []
        for propname in propnames:
            argsorted = np.argsort(relevant_props[propname].dropna())

            if propname == "area":
                candidates.extend(list(argsorted[:n_props_review])) # too low
                artifact_candidates.extend(list(argsorted[-n_props_review:])) 
            else:
                # flag props that are too low or too high
                candidates.extend(list(argsorted[:n_props_review])) # too low
                candidates.extend(list(argsorted[-n_props_review:])) # too high

        # handle if same nucleus is extreme in multiple props
        candidates = list(np.unique(candidates)) 

        # add to existing list of extreme nuclei
        candidate_nuclei = list(relevant_props.loc[candidates,"instance_id"].dropna())
        extreme_nuclei.extend(candidate_nuclei)

        # add to existing list of segmentation artifacts
        candidate_artifacts = list(relevant_props.loc[artifact_candidates,"instance_id"].dropna())
        segmentation_artifacts.extend(candidate_artifacts)

    # isolate the extreme nuclei and artifacts
    # =======================================================
    extreme_nuclei = list(np.unique(extreme_nuclei))
    segmentation_artifacts = list(np.unique(segmentation_artifacts))
    
    return Annots_DF, extreme_nuclei, segmentation_artifacts


def get_discordant_nuclei(mrcnn, regions, ignore_codes=[0,]):
    """
    Get list of nucleus instance IDs where nucleus 
    label is discordant with region labels
    """
    if 0 not in ignore_codes:
        ignore_codes.append(0)
    unique_labels = [int(j) for j in np.unique(mrcnn[..., 0]) 
                     if j not in ignore_codes]
    discordant = mrcnn[..., 1].copy()
    discordant[mrcnn[..., 0] == regions] = 0
    # ignore background, exclude, etc
    for c in ignore_codes:
        discordant[mrcnn[..., 0] == c] = 0
        if c in unique_labels:
            unique_labels.remove(c)
    # get instance ids of discordant nuclei and assign to flag
    discordant_nuclei = [int(j) for j in np.unique(discordant) if j != 0]
    return discordant_nuclei


def get_fov_stats(mrcnn, low_confidence, discordant, 
                  extreme, artifacts, 
                  roi_mask= None, keep_thresh = 0.5,
                  fov_dims= (256,256), shift_step= 128):
    """
    Gets potential FOVs along with their associated
    statistics.
    Args:
        * mrcnn [m, n, 4] - postprocessed outpt from mrcnn (4 channels)
        * low_confidence, discordant, extreme, artifacts (all [m, n]) - 
            output from get_nuclei_for_review
        * roi_mask - [m,n] where 1 indicates region inside ROI
        * keep_thresh - what fraction of FOV needs to be inside ROI
    Returns:
        df of FOV stats
    """
    print("\tGetting FOV proposal stats ...")

    # Get FOV bounds for potential FOVs for review
    M, N = mrcnn.shape[:2]
    FOV_bounds = get_fov_bounds(M, N, fov_dims=fov_dims, shift_step=shift_step)
    
    # Assume entirefield is ROI mask if not given
    fov_n_pixels = fov_dims[0] * fov_dims[1]
    if roi_mask is None:
        roi_mask = np.ones((M, N))
    
    FOV_stats = df(
        index= np.arange(len(FOV_bounds)), 
        columns= [
            "fovidx", "xmin", "xmax", "ymin" , "ymax", 
            "n_total", "predominant_label",
            "ratio_predominant", "ratio_non_predominant", 
            "n_low_confidence", "n_discordant", 
            "n_extreme", "n_artifacts",
        ])
    n_fov_proposals = len(FOV_bounds)

    # fovidx = 0; fovbounds = FOV_bounds[fovidx]
    for fovidx, fovbounds in enumerate(FOV_bounds):

        if fovidx % 500 == 0:
            print("\t\tFOV %d of %d" % (fovidx, n_fov_proposals))

        # get bounds
        (rmin, rmax, cmin, cmax) = fovbounds
        
        # check that at least part of FOV is inside ROI
        is_inside = roi_mask[rmin:rmax, cmin:cmax]
        if np.sum(is_inside[:]) / fov_n_pixels < keep_thresh:
            continue
        
        # slice
        label_fov = mrcnn[rmin:rmax, cmin:cmax, 0]
        instance_fov = mrcnn[rmin:rmax, cmin:cmax, 1]
        low_confidence_fov = low_confidence[rmin:rmax, cmin:cmax]
        discordant_fov = discordant[rmin:rmax, cmin:cmax]
        extreme_fov = extreme[rmin:rmax, cmin:cmax]
        artifact_fov = artifacts[rmin:rmax, cmin:cmax]

        # add location 
        FOV_stats.loc[fovidx, "fovidx"] = fovidx
        FOV_stats.loc[fovidx, "xmin"] = int(cmin)
        FOV_stats.loc[fovidx, "xmax"] = int(cmax)
        FOV_stats.loc[fovidx, "ymin"] = int(rmin)
        FOV_stats.loc[fovidx, "ymax"] = int(rmax)
        
        # add stats to help choose helpful FOVs for review
        unique_labels = list(np.unique(label_fov))
        unique_labels.remove(0)
                
        label_nucleus_counts = [
            len(np.unique(instance_fov[label_fov == j])) 
            for j in unique_labels]

        # handle if no relevant instances in FOV
        if len(label_nucleus_counts) < 1:
            continue
        
        n_total = np.sum(label_nucleus_counts)
        ratio_predominant = int(100 * np.max(label_nucleus_counts) / n_total)
        FOV_stats.loc[fovidx, "n_total"] = n_total
        FOV_stats.loc[fovidx, "predominant_label"] = int(unique_labels[np.argmax(label_nucleus_counts)])
        FOV_stats.loc[fovidx, "ratio_predominant"] = ratio_predominant
        FOV_stats.loc[fovidx, "ratio_non_predominant"] = 100 - ratio_predominant

        # add stats for to help choose FOVs based on lack of 
        # confidence in whether nuclei are correctly classified
        FOV_stats.loc[fovidx, "n_low_confidence"] = len(
            np.unique(instance_fov[low_confidence_fov]))
        FOV_stats.loc[fovidx, "n_discordant"] = len(
            np.unique(instance_fov[discordant_fov]))
        FOV_stats.loc[fovidx, "n_extreme"] = len(
            np.unique(instance_fov[extreme_fov]))
        FOV_stats.loc[fovidx, "n_artifacts"] = len(
            np.unique(instance_fov[artifact_fov]))
        
    return FOV_stats.dropna()


def choose_fovs_for_review(FOV_stats, im_shape, min_nuclei_per_fov= 10, 
                           fovs_per_prop= 3, exclude_edge= 128):
    """
    Select FOVs for review by choosing a mixture of representative, 
    heterogeneous, and low-confidence fov's.
    Args:
        * FOV_stats - pandas df from get_fov_stats
    Returns:
        df of fovs to be reviewed
    """
    to_exclude = []

    # exclude fovs that have too few nuclei
    to_exclude.extend(list(FOV_stats[FOV_stats["n_total"] < min_nuclei_per_fov].index))

    # exclude fovs that are too close to edge
    to_exclude.extend(list(FOV_stats[FOV_stats["xmin"] < exclude_edge].index))
    to_exclude.extend(list(FOV_stats[FOV_stats["ymin"] < exclude_edge].index))
    to_exclude.extend(list(FOV_stats[FOV_stats["xmax"] > im_shape[1] - exclude_edge].index))
    to_exclude.extend(list(FOV_stats[FOV_stats["ymax"] > im_shape[0] - exclude_edge].index))
    
    # now drop
    FOV_stats.drop(to_exclude, axis= 0, inplace= True)
    FOV_stats.reset_index(inplace=True, drop= True)

    # Now select FOVs for review based on stats
    propnames = ["ratio_predominant", "ratio_non_predominant", 
                 "n_low_confidence", "n_discordant", 
                 "n_extreme", "n_artifacts"]

    # The following scheme is meant to discourage choice of 
    # FOVs that are close to each other spatially, 
    # first we isolate the top 5% of FOV's that are
    # candidates for review based on a cetrain prop, then
    # randomly selected a subset for review
    # for example, if we have 100 fov's we keep 10 that have 
    # a high ration of predominant class nuclei, then we 
    # pick 3 of those at random for review

    n_potential_fovs = 1 + int(0.05 * FOV_stats.shape[0])
    colnames = ["fovidx", "xmin", "xmax", "ymin", "ymax", 
                "propname", "predominant_label"] + propnames
    review_fovs = df(columns= colnames)

    if FOV_stats.shape[0] > 0:
        for propidx, propname in enumerate(propnames):    

            prop_fov_idxs = list(FOV_stats.sort_values(
                propname, 0, ascending=False)[:n_potential_fovs].index)

            prop_fov_idxs = list(np.unique(np.random.choice(prop_fov_idxs, fovs_per_prop)))
            review_fovs_thisprop = FOV_stats.loc[prop_fov_idxs, :].copy()
            review_fovs_thisprop.loc[:,"propname"] = propname 

            # if an FOV was included, don't re-include
            keep_idxs = [j for j in review_fovs_thisprop.index if j not in review_fovs.index]

            # now concatenate with chosen FOVs
            review_fovs = concat((review_fovs, review_fovs_thisprop.loc[keep_idxs,:]), 
                                 axis=0, sort= False, join= 'inner') #, ignore_index=True)
    # re-arrange columns
    review_fovs = review_fovs.loc[:, colnames]
    review_fovs.reset_index(inplace= True, drop= True)
    
    return review_fovs


def choose_fovs_for_review_stratified(
    FOV_stats, im_shape, min_nuclei_per_fov= 10, 
    fovs_per_prop= 3, exclude_edge= 128, 
    relevant_labels= None):
    """
    Gets FOV's to review, making sure there is a balanced
    representation of FOV's with various predominant labels
    """
    
    unique_labels = list(np.unique(FOV_stats.loc[:, "predominant_label"]))
    if 0 in unique_labels: unique_labels.remove(0)
    
    if relevant_labels:
        unique_labels = [j for j in unique_labels if j in relevant_labels]
    
    for lblidx, lbl in enumerate(unique_labels):
        
        fov_stats_slice = FOV_stats.loc[FOV_stats["predominant_label"] == lbl, :].copy()
        
        if fov_stats_slice.shape[0] < fovs_per_prop:
            if lblidx == 0:
                review_fovs = df()
            continue
        
        # get review FOVs for this label
        print("\tChoosing FOV's for", lbl)
        review_fovs_thislbl = choose_fovs_for_review(
            FOV_stats= fov_stats_slice, 
            im_shape = im_shape,
            min_nuclei_per_fov= min_nuclei_per_fov, 
            fovs_per_prop= fovs_per_prop, 
            exclude_edge= exclude_edge,
        )
        if lblidx == 0:
            review_fovs = review_fovs_thislbl.copy()
        else:
            review_fovs = concat(
                (review_fovs, review_fovs_thislbl), 
                axis=0, sort= False, ignore_index= True)
    return review_fovs


def add_annots_and_fovs_to_db(
    Annots_DF, sqlite_save_path, review_fovs= None, 
    create_tables= False):
    """
    Adds processed nucleus annotations and chosen FOVs
    for review to sqlite database
    """
    
    # Prep SQLite database to save results
    sql = SQLite_Methods(db_path = sqlite_save_path)
    
    # get SQL formatted strings
    Annots_DF.reset_index(inplace=True, drop= True)
    Annots_sql = sql.parse_dframe_to_sql(Annots_DF, primary_key='unique_nucleus_id')
    if review_fovs is not None:
        fovs_sql = sql.parse_dframe_to_sql(review_fovs, primary_key='')
    
    # create tables if non-existent
    if create_tables:
        sql.create_sql_table(tablename='Annotations', create_str= Annots_sql['create_str'])
        if review_fovs is not None:
            sql.create_sql_table(tablename='FOVs', create_str= fovs_sql['create_str'])    
        
    # Add individual annotations
    print("\tAdding annotations to sqlite database")
    n_annots = Annots_DF.shape[0]
    for annotidx, annot in Annots_DF.iterrows():     
        if annotidx % 500 == 0:
            print("\t  Annotation %d of %d" % (annotidx, n_annots))
        with sql.conn:
            sql.update_sql_table(tablename='Annotations', entry= annot, 
                             update_str= Annots_sql['update_str'])
    # commit changes
    sql.commit_changes()
    
    if review_fovs is not None:
        # Add individual fovs
        print("\tAdding FOVs to sqlite database")
        for fovidx, fov in review_fovs.iterrows():     
            with sql.conn:
                sql.update_sql_table(tablename='FOVs', entry= fov, 
                                 update_str= fovs_sql['update_str'])
        # commit changes
        sql.commit_changes()
    
    # close
    sql.close_connection()
