import torch
import nucls_model.torchvision_detection_utils.transforms as tvdt


ISCUDA = torch.cuda.is_available()


def tensor_isin(arr1, arr2):
    r""" Compares a tensor element-wise with a list of possible values.
    See :func:`torch.isin`

    Source: https://github.com/pytorch/pytorch/pull/26144
    """
    result = (arr1[..., None] == arr2).any(-1)
    return result.type(torch.ByteTensor)


def transform_dlinput(
        tlist=None, make_tensor=True, flip_prob=0.5,
        augment_stain_sigma1=0.5, augment_stain_sigma2=0.5):
    """Transform input image data for a DL model.

    Parameters
    ----------
    tlist: None or list. If testing mode, pass as None.
    flip_prob
    augment_stain_sigma1
    augment_stain_sigma2

    """
    tmap = {
        'hflip': tvdt.RandomHorizontalFlip(prob=flip_prob),
        'augment_stain': tvdt.RandomHEStain(
            sigma1=augment_stain_sigma1, sigma2=augment_stain_sigma2),
    }
    tlist = [] if tlist is None else tlist
    transforms = []
    # go through various transforms
    for tname in tlist:
        transforms.append(tmap[tname])
    # maybe convert to tensor
    if make_tensor:
        # transforms.append(tvdt.PILToTensor(float16=ISCUDA))
        transforms.append(tvdt.PILToTensor(float16=False))
    return tvdt.Compose(transforms)
