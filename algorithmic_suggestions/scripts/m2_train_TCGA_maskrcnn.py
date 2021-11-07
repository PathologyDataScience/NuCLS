import os
import Mask_RCNN.mrcnn.model as modellib
from Mask_RCNN.mrcnn import utils

# Download COCO trained weights from Releases if needed
COCO_MODEL_PATH = os.path.join('..', '..', 'Mask_RCNN', "mask_rcnn_coco.h5")
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)
    
import algorithmic_suggestions.configs_for_AlgorithmicSuggestions_MaskRCNN as mrcnn_configs

# =================================================================
# Params

# Directory to save logs and trained model
MODEL_DIR = "/home/mohamedt/Desktop/WSI_Segmentation/Models/TCGA_maskrcnn/19July2018/"

# Which weights to start with?
init_with = "weight_file"  # imagenet, coco, last, or weight_file

if init_with == "weight_file":
    model_to_use = "nucleus20180720T1413"
    model_epoch = "mask_rcnn_nucleus_0022.h5"
    model_weights_path = "/home/mohamedt/Desktop/WSI_Segmentation/Models/TCGA_maskrcnn/19July2018/%s/%s" % (
        model_to_use, model_epoch)

augmentation = None
n_epochs = 50

# Configurations
config_train = mrcnn_configs.NucleusConfig(is_training= True)
config_train.display()

# Training dataset
dataset_train = mrcnn_configs.NucleusDataset(config= config_train)
dataset_train.load_nucleus(subset= "train")
dataset_train.prepare()

# Validation dataset
dataset_val = mrcnn_configs.NucleusDataset(config= config_train)
dataset_val.load_nucleus(subset= "val")
dataset_val.prepare()

# =================================================================
# Create model

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

# =================================================================
# Training

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
