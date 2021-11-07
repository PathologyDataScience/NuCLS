import os
import numpy as np
from pandas import Series, read_csv
from imageio import imread
import matplotlib.pylab as plt
from matplotlib.colors import ListedColormap
import hickle as hkl

from GeneralUtils import reverse_dict
from algorithmic_suggestions.bootstrapping_utils import (
    create_dir, get_shuffled_cmap
)
from algorithmic_suggestions.maskrcnn_region_integration_utils import (
    preprocess_mrcnn_output_tile, reconcile_mrcnn_with_regions, 
    get_nuclei_props_and_outliers, get_discordant_nuclei, 
    get_fov_stats, choose_fovs_for_review_stratified,
    add_annots_and_fovs_to_db,
)

# ========================================================================
# Paths and params

base_data_path = "C:/Users/tageldim/Desktop/WSI_Segmentation/Data/TCGAdataset/"
base_save_path = "/home/mohamedt/Desktop/WSI_Segmentation/Results/TCGA_maskrcnn/19July2018/nucleus20180720T1413_0030/"
imagepath = base_data_path + "TCGA_TNBC_rgbImages/ductal_normalized/"
region_mask_path = base_data_path + "TCGA_TNBC_2018-02-26/masks/core_set_151_slides/"
mrcnn_mask_path = base_save_path + "mrcnn_preds/"

mask_save_path = base_save_path + "mrcnn_regions_incorp/"
mask_vis_save_path = base_save_path + "mrcnn_regions_incorp_visualization/"
sqlite_save_path_base = base_save_path.replace('Results', 'Results_local') + "mrcnn_regions_incorp_sqlite/"
sqlite_save_path = sqlite_save_path_base + "nucleus20180720T1413_0030_regions_incorp.db"

codes_regions_path = base_data_path + "TCGA_TNBC_2018-02-26/BRCA_class_labels.tsv"

ext_imgs= ".png"

# ========================================================================
# constants

# read region codes
# This is the meaning of pixel labels in region-level output
# these codes were obtained from the "BRCA_class_labels.tsv"
# file associated with the TCGA crowdsourcing dataset
codes_region = read_csv(codes_regions_path, sep='\t', index_col= 0)

# params for getting low confidence nuclei for FOV proposal
confidence_thresh = 0.4
props_fraction = 0.05

# params for getting review fov's
fov_dims = (256,256)
shift_step = 0 
fovs_per_prop_review = 10
exclude_edge= 128
min_nuclei_per_fov = 10

# ========================================================================
# Ground work

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

# ========================================================================
# Further ground work

print("Getting list of images and some ground work ...")

# Get list of images and labels
im_list = [j.split(ext_imgs)[0] for j in os.listdir(imagepath) if ext_imgs in j]
region_prefix = "crowdsource_revision0_"
region_list = [j.split(region_prefix)[1].split(".png")[0] 
               for j in os.listdir(region_mask_path) if ".png" in j]

mrcnn_list_tiles = [j.split(".hkl")[0] for j in os.listdir(mrcnn_mask_path) if ".hkl" in j]
mrcnn_list = [str(j) for j in list(np.unique([j.split('_rowmin')[0] for j in mrcnn_list_tiles]))]

# create save path if nonexistent
create_dir(mask_save_path)
create_dir(mask_vis_save_path)
create_dir(sqlite_save_path_base)

# don't redo work
already_done = [j.split(".hkl")[0] for j in os.listdir(mask_save_path) if ".hkl" in j]
im_list = [j for j in im_list if j not in already_done]

# first we save a dataframe of what the ground truth codes mean
codes_region_df = Series(codes_region)
codes_region_df.to_csv(base_save_path + "gtruth_codes.csv")

# ========================================================================
# Define color maps

# A random colormap (for anything)
cmap = get_shuffled_cmap(plt.cm.tab20)

# random colormaps whose colors also encode labels
# We'll be using the "fixed" region codes for consistency
cmaps = {codes_region["tumor"]: get_shuffled_cmap(plt.cm.Reds),
         codes_region["fibroblast"]: get_shuffled_cmap(plt.cm.Greens),
         codes_region["lymphocyte"]: get_shuffled_cmap(plt.cm.Blues),
         codes_region["plasma_cell"]: get_shuffled_cmap(plt.cm.cool),
         codes_region["other_inflammatory"]: get_shuffled_cmap(plt.cm.autumn),
         codes_region["other"]: get_shuffled_cmap(plt.cm.Greys),
        }

# single color map to visualize single instance overlays
cmap_single = ListedColormap(["red"])

# A named color map encoding class
clist =  ['grey' for j in range(len(codes_region.keys()))]
clist[codes_region['background']] = 'black'
clist[codes_region['tumor']] = 'red'
clist[codes_region['fibroblast']] = 'green'
clist[codes_region['lymphocyte']] = 'blue'
clist[codes_region['plasma_cell']] = 'cyan'
clist[codes_region['other_inflammatory']] = 'slateblue'
clist[codes_region['necrosis_or_debris']] = 'yellow'
cmap_classlabels = ListedColormap(clist)
n_classes = len(clist)

# ========================================================================
# Iterate through unique slides

for imidx, imname in enumerate(im_list):

    print("\nslide %d of %d: %s" % (imidx, len(im_list), imname))
    
    # Read ROI
    print("\tReading RGB and region mask images ...")
    im = imread(imagepath + imname + ext_imgs, pilmode= "RGB")
    regions = imread(region_mask_path + region_prefix + imname + ".png", pilmode= "I")
    
    # Read maskrcnn output for this slide
    print("\tReading mrcnn predictions for this slide ...")
    if imname not in mrcnn_list:
        print("\t\tSlide not predicted by mrcnn! moving on!")
        continue
    
    # initialize large ROI to hold predictions from all tiles
    mrcnn = np.zeros(im.shape[:2] + (4,)) # label, instances, confidence, instance contours
    
    # get list of tiles to read
    tiles = [j for j in mrcnn_list_tiles if imname in j]
    n_tiles = len(tiles)
    
    # make sure instance assignment remains one code per instance
    instance_bias = 0
    
    for tileidx, tilename in enumerate(tiles):
        
        if tileidx % 100 == 0:
            print("\t\ttile %d of %d" % (tileidx, n_tiles-1))
        
        # read tile
        tile = hkl.load(mrcnn_mask_path + tilename + ".hkl") 
        
        # preprocess and add instance bias
        tile = preprocess_mrcnn_output_tile(tile, instance_bias= instance_bias)
        instance_bias += len(np.unique(tile[..., 1]))
        
        # get coords of tile within slide
        rowmin = int(tilename.split('rowmin')[1].split('_')[0])
        rowmax = int(tilename.split('rowmax')[1].split('_')[0])
        colmin = int(tilename.split('colmin')[1].split('_')[0])
        colmax = int(tilename.split('colmax')[1])
        
        # get existing nuclei in slide at this patch location
        patch = mrcnn[rowmin:rowmax, colmin:colmax, :] 
        patch[tile > 0] = 0
        patch = patch + tile
        
        # assign to mask
        mrcnn[rowmin:rowmax, colmin:colmax, :] = patch
        
    # Reconcile mrcnn predictions with regions
    print("\tReconciling mrcnn predictions with regions...")
    mrcnn = reconcile_mrcnn_with_regions(
        mrcnn=mrcnn, regions=regions, 
        codes_region= codes_region, 
        code_rmap_dict= code_rmap_dict, 
        KEEP_CLASSES_CODES= KEEP_CLASSES_CODES)
    
    # Get nuclei stats

    # get nucleus annotation dataframe as well as outlier nuclei
    print("\tGetting nuclei dataframe as well as outlier nuclei")
    Annots_DF, extreme_nuclei, segmentation_artifacts = get_nuclei_props_and_outliers(
        imname= imname, mrcnn= mrcnn, props_fraction= props_fraction, 
        ignore_codes= ignore_codes)
    
    # get nucleus ids of nuclei discordant with regions
    print("\tGetting discordant nuclei")
    discordant_nuclei = get_discordant_nuclei(mrcnn=mrcnn, regions=regions)
    discordant_nuclei = [j for j in discordant_nuclei if j in list(Annots_DF.index)]
    
    # Add flags for nuclei worth reviewing
    print("\tGetting outlier nuclei masks")
    
    Annots_DF.loc[:, "discordant_with_region_flag"] = 0
    Annots_DF.loc[:, "weird_shape_or_small_area_flag"] = 0
    Annots_DF.loc[:, "maybe_artifact_flag"] = 0
    
    Annots_DF.loc[discordant_nuclei, "discordant_with_region_flag"] = 1
    discordant = np.isin(mrcnn[..., 1], discordant_nuclei)
    
    Annots_DF.loc[extreme_nuclei, "weird_shape_or_small_area_flag"] = 1
    extreme = np.isin(mrcnn[..., 1], extreme_nuclei)
    
    Annots_DF.loc[segmentation_artifacts, "maybe_artifact_flag"] = 1
    artifacts = np.isin(mrcnn[..., 1], segmentation_artifacts)
    
    low_confidence = mrcnn[..., 2] < confidence_thresh
    low_confidence[mrcnn[..., 0] == 0] = 0
    
    # Divide into FOV proposals and choose fovs for review

    print("\tGetting fov stats")
    FOV_stats = get_fov_stats(
        mrcnn=mrcnn, low_confidence=low_confidence, discordant=discordant, 
        extreme=extreme, artifacts= artifacts, 
        roi_mask = regions != 0, keep_thresh= 0.5,
        fov_dims= fov_dims, shift_step= shift_step)
    
    print("\tChoosing fov's for review")
    review_fovs = choose_fovs_for_review_stratified(
        FOV_stats= FOV_stats, 
        im_shape = mrcnn.shape[:2],
        min_nuclei_per_fov= min_nuclei_per_fov, 
        fovs_per_prop= fovs_per_prop_review, 
        exclude_edge= exclude_edge,
        relevant_labels= relevant_labels_for_fov_choice,
    )
    
    # process to be relative to slide coords
    slide_name = imname.split("_")[0]
    roi_xmin = int(imname.split("_xmin")[1].split("_")[0])
    roi_ymin = int(imname.split("_ymin")[1].split("_")[0])
    review_fovs["slide_name"] = slide_name
    review_fovs["xmin"] = review_fovs["xmin"] + roi_xmin
    review_fovs["xmax"] = review_fovs["xmax"] + roi_xmin
    review_fovs["ymin"] = review_fovs["ymin"] + roi_ymin
    review_fovs["ymax"] = review_fovs["ymax"] + roi_ymin
    
    # save mrcnn reconciled-with-region mask as hickle binary
    mrcnn = mrcnn.astype(np.float32)    
    savename = "%s%s.hkl" % (mask_save_path, imname)
    print("saving", savename)
    with open(savename, 'w') as f:
        hkl.dump(mrcnn, f) 
    
    # add annotations and fovs to database
    add_annots_and_fovs_to_db(
        Annots_DF= Annots_DF, review_fovs= review_fovs, 
        sqlite_save_path= sqlite_save_path, 
        create_tables= imidx == 0)    
