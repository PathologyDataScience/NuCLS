import subprocess
import re
import numpy as np
import scipy.misc


def add_shift_steps_unbalanced(
        im_label_list_all, shift_step=0):
    """
    Appends a fixed shift step to each large image (ROI)
    Args:
        im_label_list_all - list of tuples of [(impath, lblpath),]
    Returns:
        im_label_list_all but with an added element to each tuple (shift step)
    """
    return [(j[0], j[1], shift_step) for j in im_label_list_all]


def add_shift_steps_balanced(
        im_label_list_all, fov_dims=(256, 256), 
        max_augment_factor=10):
    """
    Appends the shift step to each large image (ROI)
    to help in balancing the dataset in terms of 
    representation of ROI's of various sizes. Smaller
    ROI's get smaller shift steps (are represented by more FOV's)
    Args:
        im_label_list_all - list of tuples of [(impath, lblpath),]
    Returns:
        im_label_list_all but with an added element to each tuple (shift step)
    """
    print("getting dims to determin augment factors ...")
    imdims = []
    fovcount = []
    for imtuple in im_label_list_all:
        # get dimensions of ROI
        iminfo = str(subprocess.check_output("file " + imtuple[0], shell=True))
        N, M = re.search('(\d+) x (\d+)', iminfo).groups()
        M = int(M)
        N = int(N)
        imdims.append((M, N))
        # append approximate number of FOVs when ROI is tiled without augmentation
        fovcount.append(int(M / fov_dims[0]) * int(N / fov_dims[1]))
    
    fovcount = np.array(fovcount)
    augment_factor = np.max(fovcount) / fovcount
    # cap at max augmentation factor
    augment_factor[augment_factor > max_augment_factor] = max_augment_factor
    augment_factor = np.int32(augment_factor)
    
    print("getting ROI-specific shift step ...")
    for imidx, dim_tuple in enumerate(imdims):
        M, N = dim_tuple
        shift_step_thisim = int(min(M, N)/augment_factor[imidx])
        # shift step is at most as big as fov size
        if shift_step_thisim >= min(fov_dims[0], fov_dims[1]):
            shift_step_thisim = min(fov_dims[0], fov_dims[1]) - 1
        im_label_list_all[imidx] = (
                im_label_list_all[imidx][0], 
                im_label_list_all[imidx][1], 
                shift_step_thisim)
    
    return im_label_list_all


def get_fov_bounds(
        M, N, fov_dims=(256, 256), shift_step= 0,
        fix_size_at_edge= True):
    """
    Given an image, this get the bounds to cut it into smaller FOV's.
    Args:
    ------
        M, N - integers - image height and width
        fov_dims - x, y tuple, size of FOV's
        shift_step - if you'd like some overlap between FOV's, int
    Returns:
    --------
        FOV_bounds - list of lists, each corresponding to one FOV, 
                     in the form of [rowmin, rowmax, colmin, colmax]
    """
    # sanity checks
    assert (fov_dims[0] <= M and fov_dims[1] <= N), \
            "FOV dims must be less than image dims"
    assert (shift_step < fov_dims[0] and shift_step < fov_dims[1]), \
            "shift step must be less than FOV dims"
    
    # Needed dimensions
    m, n = fov_dims
    
    # get the bounds of of the sub-images
    Bounds_m = list(np.arange(0, M, m))
    Bounds_n = list(np.arange(0, N, n))
            
    # Add the edge
    if Bounds_m[len(Bounds_m)-1] < M:
        Bounds_m.append(M)
    if Bounds_n[len(Bounds_n)-1] < N:
        Bounds_n.append(N)
                
    # Get min and max bounds
    Bounds_m_min = Bounds_m[0:len(Bounds_m)-1]
    Bounds_m_max = Bounds_m[1:len(Bounds_m)]
    Bounds_n_min = Bounds_n[0:len(Bounds_n)-1]
    Bounds_n_max = Bounds_n[1:len(Bounds_n)]
            
    # Fix final minimum coordinate
    if fix_size_at_edge:
        if Bounds_m_min[len(Bounds_m_min)-1] > (M - m):
            Bounds_m_min[len(Bounds_m_min)-1] = M - m
        if Bounds_n_min[len(Bounds_n_min)-1] > (N - n):
            Bounds_n_min[len(Bounds_n_min)-1] = N - n
            
    #        
    # Add shifts to augment data
    #
    
    def _AppendShifted(Bounds, MaxShift):
        
        """Appends a shifted version of the bounds"""
        
        if shift_step > 0:
            Shifts = list(np.arange(shift_step, MaxShift, shift_step))
            for coordidx in range(len(Bounds)-2):
                for shift in Shifts:
                    Bounds.append((Bounds[coordidx] + shift))
                
        return Bounds
    
    # Append vertical shifts (along the m axis)
    Bounds_m_min = _AppendShifted(Bounds_m_min, m-1)
    Bounds_m_max = _AppendShifted(Bounds_m_max, m-1)
            
    # Append horizontal shifts (along the n axis)
    Bounds_n_min = _AppendShifted(Bounds_n_min, n-1)
    Bounds_n_max = _AppendShifted(Bounds_n_max, n-1)
                    
    # Initialize FOV coordinate output matrix
    num_m = len(Bounds_m_min)
    num_n = len(Bounds_n_min)
    FOV_bounds = []
    
    # Get row, col coordinates of all FOVs
    fovidx = 0
    for fov_m in range(num_m):
        for fov_n in range(num_n):
            FOV_bounds.append(\
                        [Bounds_m_min[fov_m], Bounds_m_max[fov_m], \
                        Bounds_n_min[fov_n], Bounds_n_max[fov_n]])
            fovidx += 1    
    
    return FOV_bounds


def get_imindices_str(fovbounds):
    """
    This returns a string representing the FOV indices
    Args:
    -----
        fovbounds - list of [rowmin, rowmax, colmin, colmax]
    Returns:
    --------
        string representation of indices
    """
    return "_rowmin{}".format(fovbounds[0]) + \
           "_rowmax{}".format(fovbounds[1]) + \
           "_colmin{}".format(fovbounds[2]) + \
           "_colmax{}".format(fovbounds[3])


def save_fov(im, fovbounds, savename, ext_imgs=".png", monitor_str = ""):
    """
    Given an image and fov bounds, this saves the fov.
    Args:
    ------
        im - np array
        fovbounds - list of [rowmin, rowmax, colmin, colmax]
        savename - path (including base image name), excluding extension
        ext_imgs - extension of fov
        monitor_str - identifying sting to append to beginning of print statements
    Returns:
    --------
        Nothing
    """

    # determine if rgb or grayscale
    if len(im.shape) == 3:
        is_rgb = True
    elif len(im.shape) == 2:
        is_rgb = False
    else:
        raise ValueError("Image dimensions must be 2 (grayscale) or 3 (rgb)")   
    
    
    # convert fov indices to string representation
    imindices = get_imindices_str(fovbounds)
    
    print("\t{}: Saving {}".format(monitor_str, imindices))
    
    # Convert to image
    if is_rgb:
        # RGB image
        ThisFOV = im[fovbounds[0]:fovbounds[1], \
                     fovbounds[2]:fovbounds[3], :]
        ThisFOV = scipy.misc.toimage(ThisFOV, high=np.max(ThisFOV), 
                                     low=np.min(ThisFOV))
    else:
        # Grayscale
        ThisFOV = im[fovbounds[0]:fovbounds[1], \
                     fovbounds[2]:fovbounds[3]]
        ThisFOV = scipy.misc.toimage(ThisFOV, high=np.max(ThisFOV),\
                                     low=np.min(ThisFOV), mode='I')
                        
    # save
    ThisFOV.save(savename + imindices + ext_imgs)
