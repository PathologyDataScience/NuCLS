# -*- coding: utf-8 -*-
"""
Created on Mon Aug  6 10:20:46 2018

@author: tageldim
"""

import os
import sys
sys.path.append("../")

import numpy as np
from pandas import Series, DataFrame as df, read_csv, concat
from imageio import (imread, imwrite)

# import matplotlib
# matplotlib.use('agg')
import matplotlib.pylab as plt
import matplotlib.patches as patches

from skimage.measure import regionprops

from Random_utils import (
        reverse_dict, onehottify)
from data_management import (
    get_fov_bounds, get_imindices_str)
from matplotlib.colors import ListedColormap
import hickle as hkl

from bootstrapping_utils import (
    FC_CRF, occupy_full_GT_range, create_dir, 
    get_shuffled_cmap, visualize_nuclei, 
)

from maskrcnn_region_integration_utils import (
    preprocess_mrcnn_output_tile, reconcile_mrcnn_with_regions, 
    get_nuclei_props_and_outliers, get_discordant_nuclei, 
    get_fov_stats, choose_fovs_for_review, 
    choose_fovs_for_review_stratified, 
    add_annots_and_fovs_to_db,
)

import random
import string

#%%=======================================================================
# Paths and params
#=======================================================================

base_data_path = "/mnt/Tardis/MohamedTageldin/TCGA_dataset/"
#base_data_path = "C:/Users/tageldim/Desktop/WSI_Segmentation/Data/TCGAdataset/"

base_save_path = "/home/mohamedt/Desktop/WSI_Segmentation/Results/TCGA_maskrcnn/19July2018/nucleus20180720T1413_0030/"

imagepath = base_data_path + "TCGA_TNBC_rgbImages/ductal_normalized/"
bootstrap_mask_path = base_data_path + "nucleus_segmentation_from_region_priors/"

sqlite_save_path_base = "/home/mohamedt/Desktop/WSI_Segmentation/Results_local/TCGA_maskrcnn/bootstrap_sqlite/"
sqlite_save_path = sqlite_save_path_base + "bootstrap_sqlite.db"

codes_regions_path = base_data_path + "TCGA_TNBC_2018-02-26/BRCA_class_labels.tsv"

ext_imgs= ".png"

# %%===========================================================================
# constants
# =============================================================================

# read region codes
# This is the meaning of pixel labels in region-level output
# these codes were obtained from the "BRCA_class_labels.tsv"
# file associated with the TCGA crowdsourcing dataset
codes_region = read_csv(codes_regions_path, sep='\t', index_col= 0)

# %%===========================================================================
# further ground work
# =============================================================================

# convert to dict and remove prepended "mostly_"
codes_region = dict(codes_region["GT_code"])
for k in codes_region.keys():
    if 'mostly_' in k:
        codes_region[k.replace('mostly_','')] = codes_region.pop(k)

# some processing of region code keys to reflect nucleus naming
del codes_region['roi']
del codes_region['evaluation_roi']
codes_region['background'] = 0
codes_region['fibroblast'] = codes_region.pop("stroma")
codes_region['lymphocyte'] = codes_region.pop("lymphocytic_infiltrate")
codes_region['plasma_cell'] = codes_region.pop("plasma_cells")
codes_region['other_inflammatory'] = codes_region.pop("other_immune_infiltrate")
codes_region['vascular_endothelium'] = codes_region.pop("blood_vessel")
codes_region['adipocyte'] = codes_region.pop("fat")
codes_region['skin_adnexa'] = codes_region.pop("skin_adnexia")

# label mapping -- relative to original codes
lbl_mapping = [(codes_region["angioinvasion"], codes_region["tumor"]), 
               (codes_region["dcis"], codes_region["tumor"])]

# This is the meaning of pixel labels in mrcnn output (and in bootstrap)
codes_mrcnn = {
    "background":  0, 
    "tumor": 1, 
    "fibroblast": 2, 
    "lymphocyte": 3,
    "plasma_cell": 4,
    "other_inflammatory": 5,
    "other": 6,
}

# reverse mappings
codes_region_reverse = reverse_dict(codes_region)
codes_mrcnn_reverse = reverse_dict(codes_mrcnn)

# This dict maps mrcnn nucleus codes to region codes
code_rmap_dict = dict()
for k in codes_mrcnn.keys():
    code_rmap_dict[codes_mrcnn[k]] = codes_region[k]
    
# Any class not on this list will receive its label based on
# the region in which the cell resides -- NOTE: Our reference
# here is the region map NOT the mrcnn output. This important
# because "background" in region map means it is outside ROI 
# boundary and should be removed from mrcnn prediction, whereas
# "background" in mrcnn just means mrcnn did nto predict a nucleus 
# at that location
KEEP_CLASSES = ["tumor", "fibroblast", "lymphocyte", 
                "plasma_cell", "other_inflammatory", "exclude"]
KEEP_CLASSES_CODES = [codes_region[k] for k in KEEP_CLASSES]

# relevant labels to keep in mind
# when choosing FOVs later
relevant_labels_for_fov_choice = [
    codes_region["tumor"],
    codes_region["fibroblast"],
    codes_region["lymphocyte"],
    codes_region["plasma_cell"],
    codes_region["other_inflammatory"],
]

# codes to ignore when getting nuclei for review
ignore_codes = [0, codes_region["necrosis_or_debris"]]

#%%=======================================================================
# Ground work
#=======================================================================

print("Getting list of images and some ground work ...")

# read boostrapped nuclei to visualize differences from mrcnn
bootstrap_mask_path = base_data_path + "nucleus_segmentation_from_region_priors/"
bootstrap_list = [j.split(".hkl")[0] for j in os.listdir(bootstrap_mask_path) if ".hkl" in j]

# create save path if nonexistent
create_dir(sqlite_save_path_base)

# %%===========================================================================
# Iterate through unique slides
# =============================================================================

# imidx = 0; imname = bootstrap_list[imidx] # 11 is rotated ROI
for imidx, imname in enumerate(bootstrap_list):

    print("\nslide %d of %d: %s" % (imidx, len(bootstrap_list), imname))

    # %%===========================================================================
    # Read ROI
    # =============================================================================

    print("\tReading bootstrap mask image ...")
    bootstrap = hkl.load(bootstrap_mask_path + imname + ".hkl") 

    # re-map bootstrap mrcnn to standardized codes (region)
    label_channel = bootstrap[..., 0].copy()
    for k in np.unique(label_channel):
        label_channel[label_channel == k] = code_rmap_dict[k]
    bootstrap[..., 0] = label_channel.copy()
    del label_channel

    # add "fake" certainty channel
    bootstrap = np.concatenate((bootstrap, np.random.rand(bootstrap.shape[0], bootstrap.shape[1])[..., None]), axis= -1)

    # %%===========================================================================
    # get nucleus annotation dataframe as well as outlier nuclei
    # =============================================================================

    print("\tGetting nuclei dataframe as well as outlier nuclei")
    Annots_DF, extreme_nuclei, segmentation_artifacts = get_nuclei_props_and_outliers(
        imname= imname, mrcnn= bootstrap, props_fraction= 0.1, 
        ignore_codes= ignore_codes)

    # %%===========================================================================
    # Add flags for nuclei worth reviewing
    # =============================================================================

    print("\tGetting outlier nuclei masks")

    Annots_DF.loc[:, "discordant_with_region_flag"] = 0
    Annots_DF.loc[:, "weird_shape_or_small_area_flag"] = 0
    Annots_DF.loc[:, "maybe_artifact_flag"] = 0

    Annots_DF.loc[extreme_nuclei, "weird_shape_or_small_area_flag"] = 1
    extreme = np.isin(bootstrap[..., 1], extreme_nuclei)

    Annots_DF.loc[segmentation_artifacts, "maybe_artifact_flag"] = 1
    artifacts = np.isin(bootstrap[..., 1], segmentation_artifacts)

    # %%===========================================================================
    # add annotations and fovs to database
    # =============================================================================

    add_annots_and_fovs_to_db(
        Annots_DF= Annots_DF, 
        sqlite_save_path= sqlite_save_path, 
        create_tables= imidx == 0)


