import os
from imageio import (imread, imwrite)
from algorithmic_suggestions.data_management import (
    get_fov_bounds, get_imindices_str,
)

"""
The idea here is that there are some slides/tiles that were
not used in mrcnn training because, for example:
    1- Seeding did not work well so could not get high quality bootstrapped input
    2- Too few or too many instances were detected during bootstrapping
       (but this does not negate the fact that there may be cells detected by mrcnn)

    3- These are slides that just weren't used for training for any reason.
So we get those tiles for inference to maximize size of dataset.
"""

# =======================================================================
# Paths and params

base_data_path = "C:/Users/tageldim/Desktop/WSI_Segmentation/Data/TCGAdataset/"
imagepath = base_data_path + "TCGA_TNBC_rgbImages/ductal_normalized/"
ext_imgs= ".png"

tile_save_path = base_data_path + "extra_rgb_tiles_for_mrcnn_inference/"

# so that you don't re-do work
existing_tiles_path = base_data_path + "input_for_mrcnn/images/"

# params for tiling data for maskrcnn input
fov_dims = (256, 256)
shift_step = 128

# =======================================================================
# Ground work

# Get list of images 
label_prefix = "crowdsource_revision0_"
imlist = [j.split(ext_imgs)[0] for j in os.listdir(imagepath) if ext_imgs in j]

# create save path if nonexistent
os.makedirs(tile_save_path, exist_ok=True)

# know what's done to exclude it later
already_done = [j for j in os.listdir(existing_tiles_path) if ext_imgs in j]
print("No. of tiles already done =", len(already_done))

# =======================================================================
# Iterate through unique slides and get relevant stats

for sldidx, imname in enumerate(imlist):

    print("\nslide %d of %d: %s" % (sldidx, len(imlist), imname))
    
    # read ROI
    im = imread(imagepath + imname + ext_imgs, pilmode= "RGB")
    
    # Divide into (potentially overlapping) FOVs and save

    # Get FOV bounds
    (M, N, Depth) = im.shape
    FOV_bounds = get_fov_bounds(M, N, fov_dims=fov_dims, shift_step=shift_step)
    n_fovs = len(FOV_bounds)
    
    savename_ims_base = imname.split(".")[0]
   
    for fovidx, fovbounds in enumerate(FOV_bounds):

        im_slice = im[fovbounds[0]:fovbounds[1], fovbounds[2]:fovbounds[3], ...]

        fovbounds_str = get_imindices_str(fovbounds)
        savename = savename_ims_base + fovbounds_str + ".png"
        
        if savename in already_done:
            print("\t\t", savename, "already predicted.")
        else:
            print("\tSaving", savename)
            imwrite(tile_save_path + savename, im_slice)
        
