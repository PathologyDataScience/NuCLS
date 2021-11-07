import argparse
import os
import sys
import numpy as np
import hickle as hkl
# import tensorflow as tf
from pandas import DataFrame as df
import time

# import matplotlib.pylab as plt
# %matplotlib inline

# Root directory of the project
ROOT_DIR = "/home/mohamedt/Desktop/WSI_Segmentation/Codes/mask_RCNN/"
sys.path.append(ROOT_DIR)  # To find local version of the library
sys.path.append(ROOT_DIR + "../")
sys.path.append(ROOT_DIR + "../WSI_Annotation/")

# Import Mask RCNN
import mrcnn.model as modellib

import TCGA_nucleus as nucleus 
from maskrcnn_utils_local import (
    convert_mask_to_three_channels, discard_edge_nuclei,
    add_contour_channel, add_nucleus_info_to_df)
from WSI_Annotation.SQLite_Methods import SQLite_Methods
from WSI_Annotation.general_utils import parse_coords_to_csv, list_to_str

#%% =================================================================
# Params
#=================================================================

model_to_use = "nucleus20180720T1413"
model_epoch = "mask_rcnn_nucleus_0030.h5"
model_epoch_short = model_epoch.split("_")[-1].split(".h5")[0]

#image_path = "/mnt/Tardis/MohamedTageldin/TCGA_dataset/input_for_mrcnn/images/"
image_path = "/mnt/Tardis/MohamedTageldin/TCGA_dataset/extra_rgb_tiles_for_mrcnn_inference/"

model_weights_path = "/home/mohamedt/Desktop/WSI_Segmentation/Models/TCGA_maskrcnn/19July2018/%s/%s" % (
    model_to_use, model_epoch)

# where the masks will be saved
pred_save_path = "/home/mohamedt/Desktop/WSI_Segmentation/Results/TCGA_maskrcnn/19July2018/%s_%s/" % (
    model_to_use, model_epoch_short)

# Where the sqlite database will be saved. NOTE the following: 
# This CANNOT be something mounted like tardis, otherwise database
# gets loocked and you can't do anything. see he following: 
# https://ubuntuforums.org/showthread.php?t=780891 
db_path = "/home/mohamedt/Desktop/WSI_Segmentation/Results_local/TCGA_maskrcnn/19July2018/%s_%s/" % (
    model_to_use, model_epoch_short)
db_file_path = db_path + "%s_%s.db" % (model_to_use, model_epoch_short)

logs_dir = "/home/mohamedt/Desktop/WSI_Segmentation/Models/TCGA_maskrcnn/tmp/"

# if trying to commit and database is locked
max_trial = 10
wait_seconds = 30

#%%============================================================================
# Ground work
#==============================================================================

# Parse command line arguments
# to get number og gpus and which subset to predict
# to allow running multiple scripts maskrcnn's inference only allows 1 GPU
# see: https://stackoverflow.com/questions/28549641/run-multiple-python-scripts-concurrently
parser = argparse.ArgumentParser(description="""
    Run maskrcnn inference and save to sqlite.
    Sample usage:
      python -W ignore m3_inference_TCGA_maskrcnn.py --n_gpus 3 --subset 0
    n_gpus must be the same for concurrent runs on various terminals, 
    while subset should be different so that various images are
    predicted in parallel without overlap and added to the sql database
    """)
parser.add_argument('--n_gpus', required=False,
                    metavar="n_gpus",
                    help='no of gpus for inference overall (not just this run)',
                    default= 1)
parser.add_argument('--subset', required=False,
                    metavar="subset after dividing among available gpus",
                    help="integer that is < num gpus", 
                    default= 0)
args = parser.parse_args()
n_gpus = int(args.n_gpus)
subset = int(args.subset)
assert subset < n_gpus, "subset must be between 0 and n_gpus-1"

# only print if one thread running
verbose= True #if n_gpus == 1 else False

# make sure save path exists
if not os.path.exists(pred_save_path): os.makedirs(pred_save_path)
if not os.path.exists(db_path): os.makedirs(db_path)

# sanity checks -- may decide to continue but need to be prompted
if os.path.exists(db_file_path):
    input("%s already exists. Continue?" % (db_file_path))

# get entire images list in dataset
ext = ".png"
image_ids = [j.split(ext)[0] for j in os.listdir(image_path) if ext in j]
image_ids.sort() # to make sure subsets are consistent

# Restrict to images to predict (a subset that is 1/N_GPUs of total number)
if n_gpus > 1:
    n_images_total = len(image_ids)
    subset_size = n_images_total // n_gpus
    subset_bounds = list(np.arange(0, n_images_total, subset_size))
    subset_bounds[-1] = n_images_total
    image_ids = image_ids[subset_bounds[subset]:subset_bounds[subset+1]]
    
    
#%%============================================================================
# Prep for maskRCNN inference
#==============================================================================

# Inference Configuration
config_inference = nucleus.NucleusConfig(
    is_training= False, verbose=verbose)
if verbose: config_inference.display()
    
# load dataset
dataset = nucleus.NucleusDataset(config= config_inference)
dataset.load_nucleus(specific_ids= image_ids)
dataset.prepare()

# Create model in inference mode
model = modellib.MaskRCNN(
    mode= "inference", model_dir= logs_dir, 
    config= config_inference)

# Load weights
if verbose: print("Loading weights ", model_weights_path)
model.load_weights(model_weights_path, by_name=True)

# Get bounds for which images go to what batch (just the bounds)
n_images_tot = len(dataset.image_ids)  
idx_bounds = list(np.arange(0, n_images_tot, config_inference.BATCH_SIZE))
if idx_bounds[-1] != n_images_tot:
    idx_bounds.append(n_images_tot)
    
#%%============================================================================
# Now go through batches and do inference
#==============================================================================

n_batches = len(idx_bounds) -1
for batch_idx in range(n_batches):

    # TEMP -- REMOVE ME!!! ------------------------------
    if (subset == 0) and (batch_idx < 297):
        continue
    if (subset == 1) and (batch_idx < 403):
        continue
    if (subset == 2) and (batch_idx < 2666):
        continue
    # ---------------------------------------------------

    if verbose: print("Batch %d of %d" % (batch_idx, n_batches-1))

    # Get relevant part of im_label_list_all  
    idx_start = idx_bounds[batch_idx]
    idx_end = idx_bounds[batch_idx + 1]    
    
    # load images
    imlist = []
    for image_id in dataset.image_ids[idx_start:idx_end]:
        imlist.append(dataset.load_image(image_id))
    
    # if last batch in laset subset and smaller than batch size, ignore
    if len(imlist) < config_inference.BATCH_SIZE:
        print("Batch is smaller than batch size (last batch?), won't do inference on it")
        continue

    # Detect objects in this batch of images
    r = model.detect(imlist, verbose=0)    
        
    # Init df to save coords for this batch
    Annots_DF = df(columns= [
        "unique_nucleus_id",
        "slide_name", 
        "nucleus_label", 
        "nucleus_label_confidence",
        "fov_offset_xmin_ymin", 
        "roi_offset_xmin_ymin", 
        "center_relative_to_slide_x_y",
        "bounding_box_relative_to_slide_xmin_ymin_xmax_ymax",
        "boundary_relative_to_slide_x_coords",
        "boundary_relative_to_slide_y_coords",
    ])
    
    #%% =================================================================
    # Save coords for all instances in each image in batch
    #=================================================================

    for imidx in range(len(r)):

        # Extract image info
        iminfo = dataset.image_info[idx_start:idx_end][imidx]

        # convert to three channels
        mask = convert_mask_to_three_channels(
            r[imidx]['masks'], class_ids= r[imidx]['class_ids'], 
            scores= r[imidx]['scores'])

        # discard edge nuclei
        mask = discard_edge_nuclei(mask, edge=64, keep_threshold=0.5)

        # Add contour channel
        mask = add_contour_channel(mask)

        # save mask as hickle binary
        mask = mask.astype(np.float32)    
        savename = pred_save_path + iminfo['id'] + ".hkl"
        with open(savename, 'w') as f:
            hkl.dump(mask, f) 

        # Add nucleus instance info to annotation database for this batch
        Annots_DF = add_nucleus_info_to_df(Annots_DF=Annots_DF, iminfo=iminfo, mask=mask)
        
    # if no detected instances in center, move on
    if Annots_DF.shape[0] < 1:
        print("\tno nuclei detected.")
        continue

    #%% =================================================================
    # Flush batch df into sqlite database
    #=================================================================
    
    # add annotations to database -- trying again if locked
    for trial in range(1, max_trial+1):
        try:
            # Prep SQLite database to save results
            sql = SQLite_Methods(db_path = db_file_path)
        
            # get SQL formatted strings
            Annots_sql = sql.parse_dframe_to_sql(Annots_DF, primary_key='unique_nucleus_id')
        
            # create tables if non-existent
            if batch_idx == 0:
                sql.create_sql_table(tablename='Annotations', create_str= Annots_sql['create_str'])    
                
            # Add individual annotations
            for annotidx, annot in Annots_DF.iterrows(): 
                with sql.conn:
                    sql.update_sql_table(tablename='Annotations', entry= annot, 
                                     update_str= Annots_sql['update_str'])
            # commit all changes
            sql.commit_changes()

            # close
            sql.close_connection()
            break
        
        except Exception as e:
            print("\tException:", str(e))
            if trial < max_trial: 
                print("\ttrial %d of %d: Database error? -- probably is locked" % (trial, max_trial))
                time.sleep(wait_seconds) 
            else:
                raise Exception("Tried too many times and failed to commit to database/")
        
    #%% =================================================================
    # 
    #=================================================================


