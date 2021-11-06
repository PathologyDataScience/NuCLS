# coding: utf-8

import os
import sys
sys.path.append("../")

import numpy as np
from pandas import Series
from imageio import (imread, imwrite)

import matplotlib
matplotlib.use('agg')
import matplotlib.pylab as plt

from Random_utils import (
        reverse_dict, onehottify)
from data_management import (
    get_fov_bounds, get_imindices_str)
from matplotlib.colors import ListedColormap
import hickle as hkl

from bootstrapping_utils import (
    get_nuclei_from_region_prior, FC_CRF, 
    occupy_full_GT_range, create_dir, 
    get_shuffled_cmap, visualize_nuclei, 
    )
from skimage.filters import threshold_otsu, gaussian
from scipy import ndimage
from skimage.feature import peak_local_max
from skimage.morphology import watershed
from skimage.measure import regionprops

#%%=======================================================================
# Methods
#%%=======================================================================

#%%=======================================================================
# Paths and params
#%%=======================================================================

base_data_path = "/mnt/Tardis/MohamedTageldin/TCGA_dataset/"
#base_data_path = "C:/Users/tageldim/Desktop/WSI_Segmentation/Data/TCGAdataset/"
imagepath = base_data_path + "TCGA_TNBC_rgbImages/ductal_normalized/"
labelpath = base_data_path + "TCGA_TNBC_2018-02-26/masks/core_set_151_slides/"
hematoxylinpath = base_data_path + "seeds/seed_labels/hematoxylin/"
ext_imgs= ".png"
ext_lbls= ".png"

mask_save_path = base_data_path + "nucleus_segmentation_from_region_priors/"
mask_vis_save_path = mask_save_path + "visualization/"
input_for_maskrcnn_path = base_data_path + "input_for_mrcnn/"
input_for_maskrcnn_path_images = input_for_maskrcnn_path + "images/"
input_for_maskrcnn_path_labels = input_for_maskrcnn_path + "labels/"

# %%===========================================================================
# constants
# =============================================================================

# note: we're only including classes we want to keep
# these codes were obtained from the "BRCA_class_labels.tsv"
# file associated with the TCGA crowdsourcing dataset
codes_original = {
    "background":  0, 
    "tumor": 1, 
    "stroma": 2, 
    "lymphocyte": 3,
    "plasma_cells": 10,
    "other_immune": 11,
    "other": 21,
}
codes_original_reverse = reverse_dict(codes_original)
keep_classes_original = [int(j) for j in list(codes_original_reverse.keys())]

# label mapping -- relative to original codes
# 19, 20 --> 1 (angioinvasion and DCIS map to tumor)
# 4 --> 0 (necrosis to background, necrotic nuclei get 
#         confused with lymphocytes if learned)
lbl_mapping = [(4, codes_original["background"]), 
               (19, codes_original["tumor"]), 
               (20, codes_original["tumor"])]

# New codes must use contiguous values for compatibility with one-hot encoding
# NOTE1: we shoud distinguish between "background" (0) and "other" because
# background is not learned (for example, it includes areas outside of the ROI if ROI
# is rotated, which many are). "other" on the other hand, is a learned class
# NOTE2: notice how this is a nucleus instance localization
# and classification model
codes_new = {
    "background":  0, 
    "tumor": 1, 
    "stroma": 2, 
    "lymphocyte": 3,
    "plasma_cells": 4,
    "other_immune": 5,
    "other": 6,
}
codes_new_reverse = reverse_dict(codes_new)
n_classes = len(codes_new.keys())

# This dict maps original codes to new codes for convenience
code_rmap_dict = dict()
for k in codes_original.keys():
    code_rmap_dict[codes_original[k]] = codes_new[k]

# params for tiling data for maskrcnn input
fov_dims = (256, 256)
shift_step = 128
min_n_instances = 3
max_n_instances = 10000

#%%=======================================================================
# Ground work
#%%=======================================================================

# Get list of images and labels
htx_prefix = "hematoxylin_"
label_prefix = "crowdsource_revision0_"
imlist = [j.split(ext_imgs)[0] for j in os.listdir(imagepath) if ext_imgs in j]
labellist = [j.split(label_prefix)[1].split(ext_lbls)[0]               for j in os.listdir(labelpath) if ext_lbls in j]
htxlist = [j.split(htx_prefix)[1].split(ext_lbls)[0]             for j in os.listdir(hematoxylinpath) if ext_lbls in j]

# create save path if nonexistent
create_dir(mask_save_path)
create_dir(mask_vis_save_path)
create_dir(input_for_maskrcnn_path)
create_dir(input_for_maskrcnn_path_images)
create_dir(input_for_maskrcnn_path_labels)

# don't redo work
already_done = [j.split(".hkl")[0] for j in os.listdir(mask_save_path) if ".hkl" in j]
imlist = [j for j in imlist if j not in already_done]

# Get list of image-label tuples (including hematoxylin channel)
im_label_list_all = [(imagepath + imname + ext_imgs, 
                      labelpath + label_prefix + imname + ext_lbls, 
                      hematoxylinpath + htx_prefix + imname + ext_lbls,) \
                     for imname in imlist if (imname in labellist) and (
                             imname in htxlist)]

# first we save a dataframe of what the ground truth codes mean
codes_new_df = Series(codes_new)
codes_new_df.to_csv(mask_save_path + "codes_new.csv")

#%%=======================================================================
# Define color maps
#%%=======================================================================

# A random colormap (for anything)
cmap = get_shuffled_cmap(plt.cm.tab20)

# random colormaps whose colors also encode labels
cmaps = {codes_new["tumor"]: get_shuffled_cmap(plt.cm.Reds),
         codes_new["stroma"]: get_shuffled_cmap(plt.cm.Greens),
         codes_new["lymphocyte"]: get_shuffled_cmap(plt.cm.Blues),
         codes_new["plasma_cells"]: get_shuffled_cmap(plt.cm.cool),
         codes_new["other_immune"]: get_shuffled_cmap(plt.cm.autumn),
         codes_new["other"]: get_shuffled_cmap(plt.cm.Greys),
        }

# single color map to visualize single instance overlays
cmap_single = ListedColormap(["red"])

# A named color map encoding class
clist = [
    'black', # 0- exclude
    'red', # 1- tumor
    'green', # 2- stroma / fibroblast
    'blue', # 3- lymphocyte
    'cyan', # 4- plasma cell
    'slateblue', # 5- other_immune
    'grey',  # 6- other
    ]
cmap_classlabels = ListedColormap(clist)

# %%===========================================================================
# Iterate through unique slides and get relevant stats
# =============================================================================

#sldidx = 0; sldtuple = im_label_list_all[sldidx] # 9 is rotated
for sldidx, sldtuple in enumerate(im_label_list_all):

    imname = sldtuple[0].split("/")[-1]
    print("\nslide %d of %d: %s" % (sldidx, len(im_label_list_all), imname))
    
    # %%===========================================================================
    # Read data
    # =============================================================================
    
    print("\tReading image data and initial processing ...")
    
    im = imread(sldtuple[0], pilmode= "RGB")
    regions = imread(sldtuple[1], pilmode= "I")
    htx = imread(sldtuple[2], pilmode= "L")
    
    # %%===========================================================================
    # Some label preprocessing    
    # =============================================================================
    
    # map rare classes as needed (eg "angioinvasion" --> "mostly_tumor")
    for map_tuple in lbl_mapping:
        regions[regions == map_tuple[0]] = map_tuple[1]
    
    # Everything else is mapped to "other" class
    for c in np.unique(regions):
        if int(c) not in keep_classes_original:
            regions[regions == c] = codes_original["other"]
            
    # now re-code so that codes are contiguous
    for c in np.unique(regions):
        regions[regions == c] = code_rmap_dict[c]
    
    # %%===========================================================================
    # visualize    
    # =============================================================================
    
    ## occupy full GT range
    #regions_disp = occupy_full_GT_range(regions, n_classes)
    #
    #f, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(15,15))
    #ax1.imshow(im); ax1.set_title("RGB")
    #ax2.imshow(im); 
    #ax2.imshow(regions_disp, alpha=0.4, interpolation=None, cmap=cmap_classlabels); 
    #ax2.set_title("Regions")
    
    # %%===========================================================================
    # Fully Connected Conditional Random Fields -- regions
    # This is OPTIONAL -- seems to improve things a bit
    # =============================================================================
    
    print("\tFC-CRF to Refining region boundaries with CF-CRF ...")
    
    # zero will be mapped to an extra class -- only works this way!
    # later on, we'll map it back
    pred= regions.copy()
    pred[pred == 0] = n_classes
    
    region_CRF_params = {
        "NLABELS": n_classes + 1, 
        "DELTA_COL": 10, 
        "DELTA_SIZE": 5, 
        "n_steps": 10,
        "RESIZE_FACTOR": 5, 
        "CERTAINTY": 0.8, 
        "mode": "multilabel",
    }
    regions_refined = FC_CRF(
        pred= pred, im= im, **region_CRF_params)
    del pred
    
    # what was zero in original, should be zero in refined
    regions_refined[regions_refined == n_classes] = 0
    regions_refined[regions == 0] = 0
    
    # %%===========================================================================
    # visualize    
    # =============================================================================
    
    #regions_disp = occupy_full_GT_range(regions, n_classes)
    #regions_refined_disp = occupy_full_GT_range(regions_refined, n_classes)
    #
    ## visualize result
    ## ------------------
    #f, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(15, 15))
    #ax1.imshow(regions_refined_disp, cmap= cmap_classlabels)
    #ax1.set_title('regions_refined')
    #probability_graph = ax2.imshow(regions_disp, cmap= cmap_classlabels)
    #ax2.set_title('regions')
    #plt.show()
    #
    ## visualize subregions
    ## ---------------------
    #snapshots = [(500, 500, 1500, 1500), 
    #             (1000, 1000, 2000, 2000),
    #             (2500, 4000, 3500, 5000),
    #            ]
    #for snapshot in snapshots:  
    #    (rmin, cmin, rmax, cmax) = snapshot
    #    
    #    # occupy full GT range
    #    regions_refined_slice = occupy_full_GT_range(regions_refined[rmin:rmax,cmin:cmax], n_classes)
    #    regions_slice = occupy_full_GT_range(regions[rmin:rmax,cmin:cmax], n_classes)
    #    
    #    plt.figure()
    #    f, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True)
    #    ax1.imshow(im[rmin:rmax,cmin:cmax])
    #    ax2.imshow(im[rmin:rmax,cmin:cmax])
    #    ax2.imshow(regions_refined_slice, cmap= cmap_classlabels, alpha=0.7, interpolation=None)
    #    ax3.imshow(im[rmin:rmax,cmin:cmax])
    #    ax3.imshow(regions_slice, cmap= cmap_classlabels, alpha=0.7, interpolation=None)
    
    # %%===========================================================================
    # Get nucleus sugmentation/class using region priors and hematoxylin channel    
    # =============================================================================
    
    print("\tGetting nucleus segmentation and class ...")
    
    # init nuclei semantic label channel -- init with region
    # segmentation (labels 0, 1, 2, 3 ...) and multiply 
    # by binary nucleus segmentation to convert to nucleus
    # semantic labelling. The final result is a mask
    # similar to region mask but for nuclei
    nuclei_labels_init = regions.copy()
    unique_regions = list(np.unique(regions))
    
    for c in unique_regions:
        
        if c == 0: # don't segment background
            continue
            
        nuclei_labels_init[regions == c] = c
    
        # nucleus segmentation using htx channel and region prior
        _, nuclei_hard = get_nuclei_from_region_prior(
            0 + (regions == c), htx, do_threshold= True)
        # assign label based on region prior -- everythign outside region is zero
        nuclei_thisregion = (0 + (nuclei_labels_init == c)) * nuclei_hard * c
        # superimpose
        nuclei_labels_init[nuclei_labels_init == c] = 0
        nuclei_labels_init = nuclei_labels_init + nuclei_thisregion
    
    # %%===========================================================================
    # visualize    
    # =============================================================================
    
    ## occupy full GT range
    #nuclei_disp = occupy_full_GT_range(nuclei_labels_init, n_classes)
    #
    #f, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(15, 15))
    #ax1.imshow(im); ax1.set_title("RGB")
    #ax2.imshow(im); 
    #ax2.imshow(nuclei_disp, alpha=0.7, interpolation=None, cmap=cmap_classlabels); 
    #ax2.set_title("Nuclei class")
    
    #%% clear space
    del regions
    del regions_refined
    
    # %%===========================================================================
    # Connected components and cleanup    
    # =============================================================================
    
    print("\tGetting connected components ...")
    
    try:
        min_pixels = 300 # lymphocytes are at least this 300 pixels in area
        smooth_sigma_default = 5
        smooth_sigma_lymphocyte = 2
        
        # for convenience
        unique_nuclei_labels = np.unique(nuclei_labels_init)
        
        # add a channel that will encode instance membership
        # i.e. nuclei matrix contains two channels: first one encodes label
        # while second channel encodes instance membership
        nuclei = np.zeros(nuclei_labels_init.shape + (2,))
        
        for cno in unique_nuclei_labels:
            c = codes_new_reverse[cno]
            
            if "backgr" in c:
                continue
            
            # isolate nuclei belonging to this class
            labels = 0 + (nuclei_labels_init == cno)
            
            # if there's no/negligible nuclei ignore
            if np.sum(labels) < min_pixels:
                continue
            print("\t\t%s" % (c))
            
            # Gaussian smoothing followed by otsu thresholding
            # to get rid of artifacts and small regions
            if "lymphocyte" in c:
                sigma = smooth_sigma_lymphocyte
            else:
                sigma = smooth_sigma_default
            labels = gaussian(labels, sigma=sigma, output=None, mode='nearest', preserve_range=True)
            otsu_val = threshold_otsu(labels[labels > 0])
            labels = 0 + (labels > otsu_val)
        
            # compute the exact Euclidean distance from every binary
            # pixel to the nearest zero pixel, then find peaks in this
            # distance map
            labels = labels * 255 # scipy prefers 255
            D = ndimage.distance_transform_edt(labels)
            localMax = peak_local_max(
                D, indices=False, min_distance=10, labels=labels)
            # perform a connected component analysis on the local peaks,
            # using 8-connectivity, then appy the Watershed algorithm
            markers = ndimage.label(localMax, structure=np.ones((3, 3)))[0]
            labels = watershed(-D, markers, mask=labels)
            del (D, localMax, markers)
        
            # filter very small connectd regions (artifacts)
            unique, counts = np.unique(labels, return_counts=True)
            discard = unique[counts < min_pixels]
            labels[np.isin(labels, discard)] = 0
                
            # assign to label channel after all this smoothing and cleanup
            idxs_thislbl = labels.copy()[..., None]
            idxs_thislbl = np.argwhere(idxs_thislbl > 0)
            nuclei[idxs_thislbl[..., 0], idxs_thislbl[..., 1], idxs_thislbl[..., 2]] = cno
            
            # now assign to instance mask
            # making sure instance labels are unique
            # and dont repeat any from previous classes
            max_instance = np.max(nuclei[..., 1])
            labels = labels + max_instance
            nuclei[..., 1] = nuclei[..., 1] + labels

    except MemoryError:
        print("\tMemory error encountered, moving on ...")


    # %%===========================================================================
    # visualize overall
    # =============================================================================
    
    #visualize_nuclei(im= im, instance_mask= nuclei[..., 1], labels_mask= nuclei[..., 0], 
    #                 cmaps= cmaps, figsize=(15,15))
    
    # %%===========================================================================
    # visualize upclose
    # =============================================================================
    
    #snapshots = [((3250, 3250, 3500, 3500), "tumor"),
    #             ((500, 500, 750, 750), "lymphocyte"),
    #             ((3250, 4750, 3500, 5000), "tumor"),
    #             ((2500, 4000, 2750, 4250), "necrosis"),
    #             ((3500, 1500, 3750, 1750), "stroma"),
    #             ((3500, 1500, 3750, 1750), "lymphocyte"),
    #             ((3000, 500, 3250, 750), "stroma"),
    #            ]
    #
    #for snapshot in snapshots:  
    #    (rmin, cmin, rmax, cmax) = snapshot[0]
    #    classname = snapshot[1]
    #    im_slice = im[rmin:rmax, cmin:cmax, :]
    #    lbl_slice_original = nuclei_labels_init[rmin:rmax, cmin:cmax] == codes_new[classname]
    #    
    #    thislabel = 0 + (nuclei[rmin:rmax, cmin:cmax, 0] == codes_new[classname])
    #    lbl_slice = nuclei[rmin:rmax, cmin:cmax, 1] * thislabel
    #    
    #    # use contiguous labels for better plotting
    #    lbl_disp = np.zeros(lbl_slice.shape)
    #    instances = list(np.unique(lbl_slice))
    #    for i,j in enumerate(instances):
    #        lbl_disp[lbl_slice == j] = i
    #        
    #    f, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True)
    #    ax1.imshow(im_slice); ax1.set_title('RGB')
    #    ax2.imshow(lbl_disp, cmap=cmaps[codes_new[classname]]); 
    #    ax2.set_title("%s - %d instances" % (classname, len(instances)-1)) # zero is not an instance :)
    #    ax3.imshow(lbl_slice_original, cmap=cmap); ax3.set_title("original") 
    #    plt.show()
    
    # save snapshot to visualize improvement
    # nuclei_intermediate = nuclei.copy()
    
    # %%===========================================================================
    # visualize features to vet stroma nuclei
    # =============================================================================
    
    ## NOTICE THE FOLLOWING PROBLEMS:
    ## 1- lymphocytes in stromal regions are labeled as stroma cells (fibroblasts) -- of course, since they get their labels from the region
    ## 2- clumped nuclei are detected together as one object
    ## 3- some artifacts exist (minor and few)
    ## So let's do some visualizations of features to help us handle some of these problems ...
    #
    #classname = "stroma"
    #
    #(rmin, cmin, rmax, cmax) = (3500, 1500, 3750, 1750)
    #im_slice1 = im[rmin:rmax, cmin:cmax, :]
    #thislabel = 0 + (nuclei[rmin:rmax, cmin:cmax, 0] == codes_new[classname])
    #lbl_slice1 = nuclei[rmin:rmax, cmin:cmax, 1] * thislabel
    #
    #(rmin, cmin, rmax, cmax) = (3000, 500, 3250, 750)
    #im_slice2 = im[rmin:rmax, cmin:cmax, :]
    #thislabel = 0 + (nuclei[rmin:rmax, cmin:cmax, 0] == codes_new[classname])
    #lbl_slice2 = nuclei[rmin:rmax, cmin:cmax, 1] * thislabel
    #
    ## Exctact features for all instances
    ## see: https://en.wikipedia.org/wiki/Shape_factor_(image_analysis_and_microscopy)
    ## and: http://scikit-image.org/docs/dev/api/skimage.measure.html#skimage.measure.regionprops
    #props1 = regionprops(label_image= np.int32(lbl_slice1), coordinates='rc')
    #props2 = regionprops(label_image= np.int32(lbl_slice2), coordinates='rc')
    #
    ## visualize -- props 1
    #for prop in props1:
    #    
    #    # get elongation features
    #    aspect = prop.minor_axis_length / prop.major_axis_length
    #    circularity = (4 * np.pi * prop.area) / (prop.perimeter ** 2)
    #
    #    instance_mask = np.zeros(lbl_slice1.shape)
    #    instance_mask[prop.coords[:,0], prop.coords[:,1]] = 1
    #    
    #    # mask background so you can overlay
    #    instance_mask = np.ma.masked_where(instance_mask == 0, instance_mask)
    #    
    #    plt.figure()
    #    plt.imshow(im_slice1)
    #    plt.imshow(instance_mask, alpha=0.5, interpolation=None, cmap=cmap_single)
    #    plt.title("Aspect Ratio = %.2f, circularity = %.2f, extent = %.2f, area = %.2f" % 
    #              (aspect, circularity, prop.extent, prop.area))
    #
    ## visualize -- props 2
    #for prop in props2:
    #    
    #    # get elongation features
    #    aspect = prop.minor_axis_length / prop.major_axis_length
    #    circularity = (4 * np.pi * prop.area) / (prop.perimeter ** 2)
    #
    #    instance_mask = np.zeros(lbl_slice2.shape)
    #    instance_mask[prop.coords[:,0], prop.coords[:,1]] = 1
    #    
    #    # mask background so you can overlay
    #    instance_mask = np.ma.masked_where(instance_mask == 0, instance_mask)
    #    
    #    plt.figure()
    #    plt.imshow(im_slice2)
    #    plt.imshow(instance_mask, alpha=0.5, interpolation=None, cmap=cmap_single)
    #    plt.title("Aspect Ratio = %.2f, circularity = %.2f, extent = %.2f, area = %.2f" % 
    #              (aspect, circularity, prop.extent, prop.area))
    
    # %%===========================================================================
    # Refine stromal region region nuclei classification by region props
    # =============================================================================
    
    print("\tRefining stroma region nuclei classification using region props ...")
    
    # fibroblasts have low aspect ratio and circularity, while
    # lymphocytes have high aspect ratio and circularity, 
    # everything else (i.e. mid-range) is vague/unknown
    # === UPDATE 9 July 2018 ===
    # Looking at the results, assigning lymphocytes labels
    # from stromal regions with unknown circular cells results
    # in many false positives, so getting rid of case 2, if it's
    # not clearly a fibroblast, ignore it!!
    aspect_thresh_lower = 0.4
    aspect_thresh_upper = 0.55
    circ_thresh_lower = 0.7
    circ_thresh_upper = 0.8
    classname = "stroma"
    
    # Extract features for all stroma region objects
    # see: https://en.wikipedia.org/wiki/Shape_factor_(image_analysis_and_microscopy)
    # and: http://scikit-image.org/docs/dev/api/skimage.measure.html#skimage.measure.regionprops
    thislabel = 0 + (nuclei[..., 0] == codes_new[classname])
    thislabel = nuclei[..., 1] * thislabel
    props = regionprops(label_image= np.int32(thislabel), coordinates='rc')
    
    if len(props) > 3:
        for propidx, prop in enumerate(props):
    
            if (propidx % 200) == 0:
                print("\t\tprop %d of %d" % (propidx, len(props)))
            
            try:
                if (prop.minor_axis_length < 2) or (prop.major_axis_length < 2):
                    continue
                # get elongation features
                aspect = prop.minor_axis_length / prop.major_axis_length
                circularity = (4 * np.pi * prop.area) / (prop.perimeter ** 2)
        
                # case1: this is correcty classified as a fibroblast
                if (aspect < aspect_thresh_lower) and (
                    circularity < circ_thresh_lower):
                    pass 
                
                # UPDATE 9 July 2018 -- do NOT assign lymphocyte labels!!!
                # case2: this should be classified as a lymphocyte    
                # elif (aspect > aspect_thresh_upper) and (
                #     circularity > circ_thresh_upper):        
                #     nuclei[prop.coords[:,0], prop.coords[:,1], 0] = codes_new["lymphocyte"]
                    
                # case3: this is ambiguous
                else:
                    nuclei[prop.coords[:,0], prop.coords[:,1], 0] = codes_new["background"]
        
            except ValueError:
                # by default if it raises a value error, it is likely an artifact 
                nuclei[prop.coords[:,0], prop.coords[:,1], 0] = codes_new["background"]

    # %%===========================================================================
    # Get rid of non-round objects in non-stromal regions
    # =============================================================================
    
    # Note that the standards for non-stromal regions are different
    # from stromal regions because stroma regions were a "catch all"
    # background class that contains a mixture of (mostly) fibroblasts
    # and lymphocytes. Also, fibroblasts have very elongate nuclei that 
    # whose shape properties resemble artifacts in non-stromal regions
    #
    
    print("\tGetting rid of non-round objects in non-stromal regions ...")
    
    aspect_thresh_lower = 0.5
    circ_thresh_lower = 0.6
    extent_thresh_upper = 0.8 # very large extent means it looks like a rectangle
    
    for cno in unique_nuclei_labels:
        
        classname = codes_new_reverse[cno]
        if ("backg" in classname) or ("stroma" in classname):
            continue
        print("\t\t=== %s ===" % (classname))
    
        thislabel = 0 + (nuclei[..., 0] == codes_new[classname])
        thislabel = nuclei[..., 1] * thislabel
        props = regionprops(label_image= np.int32(thislabel), coordinates='rc')
        if len(props) < 3:
            continue
        
        for propidx, prop in enumerate(props):  
    
            if (propidx % 200) == 0:
                print("\t\tprop %d of %d" % (propidx, len(props)))
    
            try:
                if (prop.minor_axis_length < 2) or (prop.major_axis_length < 2):
                    continue
                # get elongation features
                aspect = prop.minor_axis_length / prop.major_axis_length
                circularity = (4 * np.pi * prop.area) / (prop.perimeter ** 2)
        
                # this is ambiguous / probably an artifact
                if (aspect < aspect_thresh_lower) and (
                    circularity < circ_thresh_lower) and (
                    prop.extent > extent_thresh_upper):
                    nuclei[prop.coords[:,0], prop.coords[:,1], 0] = codes_new["background"]
        
            except ValueError:
                # by default if it raises a value error, it is likely an artifact 
                nuclei[prop.coords[:,0], prop.coords[:,1], 0] = codes_new["background"]

    # %%===========================================================================
    # Final processing
    # =============================================================================

    print("\tfinal processing ...")
    
    min_pixels = 300 # lymphocytes are typically > 300 pixels in area
    tmp_labels = nuclei[..., 0].copy()
    tmp_instances = nuclei[..., 1].copy()
    
    # filter very small connectd regions (artifacts)
    unique, counts = np.unique(tmp_instances, return_counts=True)
    discard = unique[counts < min_pixels]
    tmp_instances[np.isin(tmp_instances, discard)] = 0
    tmp_labels[np.isin(tmp_labels, discard)] = 0
    
    # there is no such thing as a background instance
    tmp_instances[tmp_labels == 0] = 0
    
    # assign back to mask
    nuclei[..., 0] = tmp_labels.copy()
    nuclei[..., 1] = tmp_instances.copy()
    
    # cleanup
    del tmp_labels
    del tmp_instances
    
    # %%===========================================================================
    # visualize overall + save visualization
    # =============================================================================
    
    print("\tvisualizing and saving ...")
    savename = mask_vis_save_path + imname.split(".")[0] + ".tif"
    visualize_nuclei(im= im, instance_mask= nuclei[..., 1], labels_mask= nuclei[..., 0], 
                     cmaps= cmaps, alpha_im= 0.7, alpha_nuclei= 0.7, figsize=(15,15), 
                     savename= savename, show= False)
    
    # %%===========================================================================
    # visualize upclose
    # =============================================================================
    
    #snapshots = [((3250, 3250, 3500, 3500), "tumor"),
    #             ((500, 500, 750, 750), "lymphocyte"),
    #             ((3250, 4750, 3500, 5000), "tumor"),
    #             ((2500, 4000, 2750, 4250), "necrosis"),
    #             ((3500, 1500, 3750, 1750), "stroma"),
    #             ((3500, 1500, 3750, 1750), "lymphocyte"),
    #             ((3000, 500, 3250, 750), "stroma"),
    #            ]
    #
    #for snapshot in snapshots:  
    #    (rmin, cmin, rmax, cmax) = snapshot[0]
    #    classname = snapshot[1]
    #    im_slice = im[rmin:rmax, cmin:cmax, :]
    #    lbl_slice_original = nuclei_labels_init[rmin:rmax, cmin:cmax] == codes_new[classname]
    #    
    #    thislabel = 0 + (nuclei[rmin:rmax, cmin:cmax, 0] == codes_new[classname])
    #    lbl_slice = nuclei[rmin:rmax, cmin:cmax, 1] * thislabel
    #    
    #    thislabel = 0 + (nuclei_intermediate[rmin:rmax, cmin:cmax, 0] == codes_new[classname])
    #    lbl_slice_interm = nuclei_intermediate[rmin:rmax, cmin:cmax, 1] * thislabel
    #    
    #    # use contiguous labels for better plotting -- new
    #    lbl_disp = np.zeros(lbl_slice.shape)
    #    instances = list(np.unique(lbl_slice))
    #    for i,j in enumerate(instances):
    #        lbl_disp[lbl_slice == j] = i
    #        
    #    # use contiguous labels for better plotting -- intermediate
    #    lbl_disp_interm = np.zeros(lbl_slice_interm.shape)
    #    instances_interm = list(np.unique(lbl_slice_interm))
    #    for i,j in enumerate(instances_interm):
    #        lbl_disp_interm[lbl_slice_interm == j] = i
    #    
    #    f, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, sharey=True)
    #    ax1.imshow(im_slice); ax1.set_title('RGB')
    #    ax2.imshow(lbl_disp, cmap=cmaps[codes_new[classname]]); 
    #    ax2.set_title("final\n%d instances" % (len(instances)-1)) # zero is not an instance :)
    #    ax3.imshow(lbl_disp_interm, cmap=cmaps[codes_new[classname]]); 
    #    ax3.set_title("interm\n%d instances" % (len(instances_interm)-1)) # zero is not an instance :)
    #    ax4.imshow(lbl_slice_original, cmap=cmap); ax4.set_title("original") 
    #    plt.suptitle(classname, fontsize=14, y=0.8)
    #    plt.show()
    
    # %%===========================================================================
    # Save mask
    # =============================================================================
    
    print("\tSaving mask as hickle binary ...")
    
    # convert to int32 for memory efficiency
    nuclei = nuclei.astype(np.int32)
    
    # see: https://github.com/telegraphic/hickle
    savename = mask_save_path + imname.split(".")[0] + ".hkl"
    with open(savename, 'w') as f:
        hkl.dump(nuclei, f) 
    
    # %%===========================================================================
    # Load mask
    # =============================================================================
    
    # print("\tLoding mask from hickle binary ...")
    
    # # see: https://github.com/telegraphic/hickle
    # savename = mask_save_path + imname.split(".")[0] + ".hkl"
    # nuclei = hkl.load(savename) 
    
    # %%===========================================================================
    # Show concept behind overlapping tiles
    # =============================================================================
    
    ## The idea is to save tiles that overlap horizontally and vertically, so that:
    ## During training: this acts augmentation
    ## During inference: only keep predictions from the center of each tile so that
    ## nuclei that lie at the edge between various tiles don't get misclassified.
    ## in other words, nuclei get their predictions from tiles in which they are
    ## located centrally
    #
    #tilingfig = "../ideas/tiling.png"
    #plt.figure(figsize=(15,15))
    #plt.imshow(plt.imread(tilingfig))
    
    # %%===========================================================================
    # Divide into (potentially overlapping) FOVs and save
    # =============================================================================
    
    # Get FOV bounds
    (M, N, Depth) = im.shape
    FOV_bounds = get_fov_bounds(M, N, fov_dims=fov_dims, shift_step=shift_step)
    n_fovs = len(FOV_bounds)
    
    savename_ims_base = input_for_maskrcnn_path_images + imname.split(".")[0]
    savename_masks_base = input_for_maskrcnn_path_labels + imname.split(".")[0]
   
    # size threshold for exclusion (edge of tile)
    min_pixels = 150

    # fovidx = 0; fovbounds = FOV_bounds[fovidx]
    for fovidx, fovbounds in enumerate(FOV_bounds):

        # slice
        im_slice = im[fovbounds[0]:fovbounds[1], fovbounds[2]:fovbounds[3], ...]
        mask_slice = nuclei[fovbounds[0]:fovbounds[1], fovbounds[2]:fovbounds[3], ...]

        # this will be used for convenience
        tmp_labels = mask_slice[..., 0]
        tmp_instances = mask_slice[..., 1]

        # filter very small connectd regions -- since we just
        # sliced the mask, there are gonna be artifacts at edge
        # of each tile
        unique_instances, counts = np.unique(tmp_instances, return_counts= True)
        discard = unique_instances[counts < min_pixels]
        tmp_instances[np.isin(tmp_instances, discard)] = 0
        tmp_labels[np.isin(tmp_labels, discard)] = 0
        
        # get unique nuclei instances (after initial filtering)
        unique_instances, counts = np.unique(tmp_instances, return_counts= True)
        unique_instances = list(unique_instances)
        if 0 in unique_instances:
            unique_instances.remove(0)
        n_instances = len(unique_instances)
        print("\tfov %d of %d: %d instances" %(fovidx, n_fovs, n_instances))

        # Ignore masks that have < min_n_instances
        if n_instances < min_n_instances:
            print("\t\tToo few instances. FOV ignored.")
            continue
        
        # Cap to max_n_instances
        # this is needed to prevent memory issues when mask
        # is loaded into GPU memory. This is OK, since
        # our tiles are overlapping and we randomly
        # choose what instances to keep, so if something
        # is discarded from one tile, it may be kept in its
        # overlapping neighbor
        if n_instances > max_n_instances:
            
            # shuffle so instances kept aren't all in one corner
            np.random.shuffle(unique_instances)
            
            # now remove extra instances
            discard = unique_instances[max_n_instances-1:]
            print("\t\tRemoving %d extra nuclei instances" % (len(discard)))
            
            tmp_instances[np.isin(tmp_instances, discard)] = 0
            tmp_labels[np.isin(tmp_labels, discard)] = 0
    
        # assign back to mask
        mask_slice[..., 0] = tmp_labels.copy()
        mask_slice[..., 1] = tmp_instances.copy()
        del tmp_labels
        del tmp_instances
        
        # save rgb
        fovbounds_str = get_imindices_str(fovbounds)
        savename = savename_ims_base + fovbounds_str + ".png"
        imwrite(savename, im_slice)
        
        # save mask
        # see: https://github.com/telegraphic/hickle
        savename = savename_masks_base + fovbounds_str + ".hkl"
        with open(savename, 'w') as f:
            hkl.dump(mask_slice, f) 
        
        # # load mask (confirm it saved correctly)
        # tmp = hkl.load(savename) 
        # plt.figure(); plt.imshow(tmp[..., 0], cmap=cmap)
        # plt.figure(); plt.imshow(tmp[..., 1], cmap=cmap)
    
