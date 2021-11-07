# coding: utf-8
import os
import sys

# Root directory of the project
ROOT_DIR = "/home/mohamedt/Desktop/WSI_Segmentation/Codes/mask_RCNN"

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
import mrcnn.model as modellib
#from mrcnn.config import Config
from mrcnn import utils
#from mrcnn import visualize
#from mrcnn.model import log

# other imports
#import numpy as np
#import cv2
#import matplotlib
#import matplotlib.pyplot as plt

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)
    
import TCGA_nucleus as nucleus # UNCOMMENT ME !!!

#%% =================================================================
# Params
#=================================================================

# Directory to save logs and trained model
MODEL_DIR = "/home/mohamedt/Desktop/WSI_Segmentation/Models/TCGA_maskrcnn/19July2018/"

# Which weights to start with?
init_with = "weight_file"  # imagenet, coco, last, or weight_file

if init_with == "weight_file":
    model_to_use = "nucleus20180720T1413"
    model_epoch = "mask_rcnn_nucleus_0022.h5"
    model_weights_path = "/home/mohamedt/Desktop/WSI_Segmentation/Models/TCGA_maskrcnn/19July2018/%s/%s" % (
        model_to_use, model_epoch)

## Image augmentation
## http://imgaug.readthedocs.io/en/latest/source/augmenters.html
#augmentation = iaa.SomeOf((0, 2), [
#    iaa.Fliplr(0.5),
#    iaa.Flipud(0.5),
#    iaa.OneOf([iaa.Affine(rotate=90),
#               iaa.Affine(rotate=180),
#               iaa.Affine(rotate=270)]),
#    iaa.Multiply((0.8, 1.5)),
#    iaa.GaussianBlur(sigma=(0.0, 5.0))
#])
augmentation = None

n_epochs = 50

#%% =================================================================
# Configurations
#=================================================================

config_train = nucleus.NucleusConfig(is_training= True)
config_train.display()

#%% =================================================================
# Dataset
#=================================================================

# Training dataset
dataset_train = nucleus.NucleusDataset(config= config_train)
dataset_train.load_nucleus(subset= "train")
dataset_train.prepare()

# Validation dataset
dataset_val = nucleus.NucleusDataset(config= config_train)
dataset_val.load_nucleus(subset= "val")
dataset_val.prepare()

# # Load and display random samples
# image_ids = np.random.choice(dataset_train.image_ids, 4)
# for image_id in image_ids:
#     image = dataset_train.load_image(image_id)
#     mask, class_ids = dataset_train.load_mask(image_id)
#     visualize.display_top_masks(image, mask, class_ids, dataset_train.class_names)

#%% =================================================================
# Create model
#=================================================================

# Create model in training mode
model = modellib.MaskRCNN(mode="training", config= config_train,
                          model_dir=MODEL_DIR)

if init_with == "imagenet":
    model.load_weights(model.get_imagenet_weights(), by_name=True)

elif init_with == "coco":
    # Load weights trained on MS COCO, but skip layers that
    # are different due to the different number of classes
    # See README for instructions to download the COCO weights
    model.load_weights(COCO_MODEL_PATH, by_name=True,
                       exclude=["mrcnn_class_logits", "mrcnn_bbox_fc", 
                                "mrcnn_bbox", "mrcnn_mask"])
elif init_with == "last":
    # Load the last model you trained and continue training
    model.load_weights(model.find_last(), by_name=True)

elif init_with == "weight_file":
    print("Loading weights from", model_weights_path)
    model.load_weights(model_weights_path, by_name=True)

#%% =================================================================
# Training
#=================================================================

# Train in two stages:
# 1- Only the heads. Here we're freezing all the backbone layers and training 
#    only the randomly initialized layers (i.e. the ones that we didn't use 
#    pre-trained weights from MS COCO). To train only the head layers, 
#    pass layers='heads' to the train() function.
# 2- Fine-tune all layers. For this simple example it's not necessary, 
#    but we're including it to show the process. Simply pass layers="all 
#    to train all layers.

# Train the head branches
# Passing layers="heads" freezes all layers except the head
# layers. You can also pass a regular expression to select
# which layers to train by name pattern.
print("Training network heads ...")
model.train(dataset_train, dataset_val, 
            learning_rate= config_train.LEARNING_RATE, 
            epochs= n_epochs,
            augmentation= augmentation,
            layers='heads')

