import os
import sys
import numpy as np
import hickle as hkl

from Mask_RCNN.mrcnn.config import Config
from Mask_RCNN.mrcnn import utils
from GeneralUtils import reverse_dict, AllocateGPU

# =================================================================
#  Paths

if os.getlogin() == 'tageldim':
    is_laptop = True
else:
    is_laptop = False

# Path to trained weights file
COCO_WEIGHTS_PATH = os.path.join('..', 'Mask_RCNN', "mask_rcnn_coco.h5")

# Data directories    
# DEFAULT_LOGS_DIR is the directory to save logs and model checkpoints, 
# if not provided through the command line argument --logs
if is_laptop:
    DEFAULT_LOGS_DIR = "C:/Users/tageldim/Desktop/WSI_Segmentation/tmp/"
    input_for_maskrcnn_path = "C:/Users/tageldim/Desktop/WSI_Segmentation/Data/TCGAdataset/input_for_mrcnn/"
    imagepath = input_for_maskrcnn_path + "images/"
    labelpath = input_for_maskrcnn_path + "labels/"
else:
    DEFAULT_LOGS_DIR = "/home/mohamedt/Desktop/WSI_Segmentation/tmp/"
    input_for_maskrcnn_path = "C:/Users/tageldim/Desktop/TCGA_dataset/extra_rgb_tiles_for_mrcnn_inference/"
    imagepath = input_for_maskrcnn_path
    
# =================================================================
#  Dataset Configurations 

class DatasetConfigs(object):
    
    def __init__(self, is_training= True, verbose= False):
        
        self.is_training = is_training
    
        #=============================================================
        # Gtruth encoding, number of classes, and extensions
        
        # Class names and ground truth encoding
        # background is already assumed to be zero
        # i.e. do NOT include background class. 
        # Also, class IDs have to be contiguous
        self.GTRUTH_ENCODING = {
            "tumor": 1, 
            "stroma": 2, 
            "lymphocyte": 3,
            "plasma_cells": 4,
            "other_immune": 5,
            "necrosis": 6, 
            "other": 7,
        }
        self.GTRUTH_ENCODING_REVERSE = reverse_dict(self.GTRUTH_ENCODING)
        self.EXT_IMGS = ".png"
        self.EXT_LBLS = ".hkl"
        
        #=============================================================
        # image IDs and list (if training mode)
        
        if self.is_training:
            # substrings that define validation set
            self.val_id_substrings = ["A2-A3XY",]
            # self.val_id_substrings = ["A2-A0T0",]
            
            # image IDs -- all
            if verbose: print("Getting image ID's ...")
            self.IMAGE_IDS = [j.split(self.EXT_IMGS)[0] for j in os.listdir(imagepath) 
                         if self.EXT_IMGS in j]
            self.LABEL_IDS = [j.split(self.EXT_LBLS)[0] for j in os.listdir(labelpath) 
                         if (self.EXT_LBLS in j)]
            assert len(self.IMAGE_IDS) == len(self.LABEL_IDS)
            
            # shuffle imageIDs for better training
            self.IMAGE_IDS = np.array(self.IMAGE_IDS)
            np.random.shuffle(self.IMAGE_IDS)
            self.IMAGE_IDS = list(self.IMAGE_IDS)
            
            # take out the validation set
            self.IMAGE_IDS_VAL = []
            for j in self.IMAGE_IDS:
                for substr in self.val_id_substrings:
                    if substr in j:
                        self.IMAGE_IDS_VAL.append(j)
                        self.IMAGE_IDS.remove(j)
            if verbose:
                print("n_training =", len(self.IMAGE_IDS)) 
                print("n_validation =", len(self.IMAGE_IDS_VAL)) 
        else:
            self.IMAGE_IDS = None
            self.IMAGE_IDS_VAL = None

# =================================================================
#  Configurations for training / inference

class NucleusConfig(Config):
    """
    Configuration for the nucleus segmentation dataset.
    """    
    
    # General maskrcnn training configurations
    
    LEARNING_RATE = 1e-4
        
    # Maximum number of ground truth instances to use in one image
    MAX_GT_INSTANCES = 30 # 30 works

    # Max number of final detections per image
    DETECTION_MAX_INSTANCES = 200 # 400 # 200 works

    # Don't exclude based on confidence. Since we have two classes
    # then 0.5 is the minimum anyway as it picks between nucleus and BG
    DETECTION_MIN_CONFIDENCE = 0

    # Backbone network architecture
    # Supported values are: resnet50, resnet101
    BACKBONE = "resnet50"

    # Input image resizing
    IMAGE_MIN_DIM = 128 
    IMAGE_MAX_DIM = 128 
    IMAGE_MIN_SCALE = 2.0

    # Length of square anchor side in pixels
    RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)

    # ROIs kept after non-maximum supression (training and inference)
    POST_NMS_ROIS_TRAINING = 500 #1000 # 500 works
    POST_NMS_ROIS_INFERENCE = 1000 #2000 # 1000 works

    # How many anchors per image to use for RPN training
    RPN_TRAIN_ANCHORS_PER_IMAGE = 32 #64

    # Image mean (RGB)
    MEAN_PIXEL = np.array([43.53, 39.56, 48.22])

    # If enabled, resizes instance masks to a smaller size to reduce
    # memory load. Recommended when using high-resolution images.
    USE_MINI_MASK = True
    MINI_MASK_SHAPE = (32, 32)  # (height, width) of the mini-mask

    # Number of ROIs per image to feed to classifier/mask heads
    # The Mask RCNN paper uses 512 but often the RPN doesn't generate
    # enough positive proposals to fill this and keep a positive:negative
    # ratio of 1:3. You can increase the number of proposals by adjusting
    # the RPN NMS threshold.
    TRAIN_ROIS_PER_IMAGE = 64 # 128 # 64 works  
    
    def __init__(self, is_training= True, verbose= True):
        """
        Override parent __init__()
        """
        # Give the configuration a recognizable name
        self.NAME = "nucleus"
        
        # instantiate dataset configurations 
        self.is_training = is_training
        self.dataset_configs = DatasetConfigs(is_training= self.is_training)
        
        # batch size and the what nots

        # Number of classes (including background)
        self.NUM_CLASSES = len(self.dataset_configs.GTRUTH_ENCODING.keys()) + 1

        if self.is_training:
            # NUMBER OF GPUs to use. For CPU training, use 1
            self.GPU_COUNT = 2 #1
            self.GPUs_to_use = [2,3] # use "None" to pick by lowest memory
            AllocateGPU(self.GPU_COUNT, GPUs_to_use= self.GPUs_to_use, verbose= verbose)
            self.IMAGES_PER_GPU = 8
            
            # Number of training and validation steps per epoch
            self.STEPS_PER_EPOCH = (len(self.dataset_configs.IMAGE_IDS)) // self.IMAGES_PER_GPU
            self.VALIDATION_STEPS = max(1, len(self.dataset_configs.IMAGE_IDS_VAL) // self.IMAGES_PER_GPU)
            
            # Random crops of size IMAGE_MIN_DIMxIMAGE_MAX_DIM
            self.IMAGE_RESIZE_MODE = "crop"
            
            # Non-max suppression threshold to filter RPN proposals.
            # You can increase this during training to generate more propsals.
            self.RPN_NMS_THRESHOLD = 0.9 # 0.9 works
       
        else: # *** i.e. INFERENCE MODE *** #
            
            # Inference batch size
            self.GPU_COUNT = 1 # MUST be 1        
            self.GPUs_to_use = [3,] # use "None" to pick by lowest memory
            AllocateGPU(self.GPU_COUNT, GPUs_to_use= self.GPUs_to_use, verbose= verbose)
            self.IMAGES_PER_GPU = 8   
            
            # The folowing DOES NOT MATTER (i.e. will be ignored)
            self.STEPS_PER_EPOCH = 10
            self.VALIDATION_STEPS = 10
            
            # Don't resize imager for inferencing (override)
            self.IMAGE_RESIZE_MODE = "pad64"
            # Non-max suppression threshold to filter RPN proposals.
            # You can increase this during training to generate more propsals.
            self.RPN_NMS_THRESHOLD = 0.7
   
        # Now set values of computed attributes (from parent's __init__)
        # Effective batch size
        self.BATCH_SIZE = self.IMAGES_PER_GPU * self.GPU_COUNT

        # Input image size
        if self.IMAGE_RESIZE_MODE == "crop":
            self.IMAGE_SHAPE = np.array([self.IMAGE_MIN_DIM, self.IMAGE_MIN_DIM, 3])
        else:
            self.IMAGE_SHAPE = np.array([self.IMAGE_MAX_DIM, self.IMAGE_MAX_DIM, 3])

        # Image meta data length
        # See compose_image_meta() for details
        self.IMAGE_META_SIZE = 1 + 3 + 3 + 4 + 1 + self.NUM_CLASSES
    

class NucleusDataset(utils.Dataset):
    
    def __init__(self, config):
        
        # Instantiate parent class
        super().__init__()
        
        # fetch relevant attribs from config object
        self.GTRUTH_ENCODING = config.dataset_configs.GTRUTH_ENCODING
        self.GTRUTH_ENCODING_REVERSE = config.dataset_configs.GTRUTH_ENCODING_REVERSE
        self.IMAGE_IDS = config.dataset_configs.IMAGE_IDS
        self.IMAGE_IDS_VAL = config.dataset_configs.IMAGE_IDS_VAL
        self.EXT_IMGS = config.dataset_configs.EXT_IMGS
        self.EXT_LBLS = config.dataset_configs.EXT_LBLS
        self.MAX_GT_INSTANCES = config.MAX_GT_INSTANCES
        
    def load_nucleus(self, dataset_dir= None, subset= None, 
                     specific_ids= None):
        """
        Load a subset of the nuclei dataset.
        dataset_dir: Root directory of the dataset
        subset: Subset to load. one of:
                * train: training set
                * val: validation set
        specific_ids: If you want to provide the image ids externally
            
         if not specific_ids, either dataset_dir or subset but 
         not both (XOR) must be specified.    
        """
        
        if specific_ids:
            image_ids = specific_ids
        
        else:
            # either dataset_dir or subset but not both (XOR)
            assert dataset_dir or subset
            assert not (dataset_dir and subset)
            assert subset in [None, "train", "val"]

            # image ids to use
            if subset == "train":
                image_ids = self.IMAGE_IDS
            elif subset == "val":
                image_ids = self.IMAGE_IDS_VAL
            else:
                # Get image ids from directory
                image_ids = [
                    j.split(self.EXT_IMGS)[0] for j in os.listdir(dataset_dir) 
                    if self.EXT_IMGS in j]

        # Add classes. Naming the dataset nucleus
        for k in range(1, len(self.GTRUTH_ENCODING.keys())+1):
            self.add_class(source= "nucleus", class_id= k, 
                           class_name= self.GTRUTH_ENCODING_REVERSE[k])
        # Add images
        for image_id in image_ids:
            self.add_image(
                "nucleus", image_id= image_id,
                path= imagepath + image_id + self.EXT_IMGS)
    
    def load_mask(self, image_id):
        """Generate instance masks for an image.
       Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        
        true_id = self.image_info[image_id]["id"]
        
        # Read mask file (hkl)
        mask_loaded = hkl.load(labelpath + true_id + self.EXT_LBLS) 

        unique_instances = list(np.unique(mask_loaded[..., 1]))
        unique_instances.remove(0)
        
        # cap number of instances to maximum
        if len(unique_instances) > self.MAX_GT_INSTANCES:
            np.random.shuffle(unique_instances)
            unique_instances = unique_instances[:self.MAX_GT_INSTANCES]
        
        n_instances = len(unique_instances)
        mask = np.zeros((mask_loaded.shape[:2] + (n_instances,)))
        labels = np.zeros((n_instances,))

        for pos, instanceid in enumerate(unique_instances):
            mask[..., pos] = mask_loaded[..., 1] == instanceid
            position = np.argwhere(mask[..., pos])[0]
            labels[pos] = mask_loaded[position[0], position[1], 0]

        mask = mask.astype(np.bool)
        labels = labels.astype(np.int32)
        
        return mask, labels
    
    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "nucleus":
            return info["id"]
        else:
            super(self.__class__, self).image_reference(image_id)


#=================================================================
    
if __name__ == '__main__':
    pass
