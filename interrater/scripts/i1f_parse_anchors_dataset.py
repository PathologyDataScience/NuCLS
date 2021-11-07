from os.path import join as opj
import numpy as np
from typing import Union
from imageio import imwrite
from skimage.transform import resize
from pandas import DataFrame, read_sql_query, read_csv, Series
from histomicstk.annotations_and_masks.annotations_to_masks_handler import \
    _visualize_annotations_on_rgb
from histomicstk.annotations_and_masks.annotations_to_object_mask_handler \
    import contours_to_labeled_object_mask

from configs.nucleus_style_defaults import Interrater as ir, \
    DefaultAnnotationStyles as das
from interrater.interrater_utils import _maybe_mkdir, \
    _connect_to_anchor_db, _get_rgb_for_interrater

# NOTE: CandygramAPI is a private API that connects to a girder client
# The class contains private access tokens that cannot be shared publicly.
from configs.ACS_configs import CandygramAPI  # noqa


def _get_and_maybe_update_fovmeta(dbcon):
    """"""
    # get fov metadata
    fovmetas = read_sql_query(f"""
        SELECT  * FROM "fov_meta"
    ;""", dbcon)

    cols = list(fovmetas.columns)

    if 'FOVID' not in cols:
        # Assign a unique fov ID to link to dataset RGBs and masks, which
        # will NOT have the exact same xmin, ymin etc since the minimum
        # bbox around their annotations will be different
        cols.insert(1, 'FOVID')
        fovmetas.loc[:, 'FOVID'] = [f'ANCHFOV-{i}' for i in fovmetas.index]
        fovmetas = fovmetas.loc[:, cols]

        # save to disk
        fovmetas.to_sql(
            name=f'fov_meta', con=dbcon,
            if_exists='replace', index=False)

    return fovmetas


def _adjust_coordstr(coordstr: str, offset: int, sf: int = 1):
    return ','.join([
        str(int(sf * (int(j) + offset))) for j in coordstr.split(',')])


def _maybe_coordstr(row):
    if row['has_algorithmic_boundary'] == 0:
        row['coords_x'] = ','.join([
            str(j) for j in [
                row['xmin'], row['xmax'], row['xmax'],
                row['xmin'], row['xmin'],
            ]
        ])
        row['coords_y'] = ','.join([
            str(j) for j in [
                row['ymin'], row['ymin'], row['ymax'],
                row['ymax'], row['ymin'],
            ]
        ])
    else:
        # IMPORTANT!!! make sure x/y min/max correpond to coords x/y
        for c in ('x', 'y'):
            coo = [int(j) for j in row[f'coords_{c}'].split(',')]
            row[f'{c}min'], row[f'{c}max'] = min(coo), max(coo)

    return row


def _get_and_fix_contours(
        dbcon, whoistruth: str, evalset: str, fmeta: Union[dict, Series]):
    maxsize = 125
    # get anchors
    truthcol = ir._get_truthcol(whoistruth=whoistruth, unbiased=False)
    contours = read_sql_query(f"""
        SELECT "anchor_id", "{truthcol}" AS "group"
             , "xmin", "ymin", "xmax", "ymax"
             , "has_algorithmic_boundary"
             , "algorithmic_coords_x" AS "coords_x"
             , "algorithmic_coords_y" AS "coords_y"
        FROM "v3.1_final_anchors_{evalset}_{whoistruth}_AreTruth"
        WHERE "fovname" = "{fmeta['fovname']}"
    ;""", dbcon)

    if contours.shape[0] < 1:
        return contours, {}

    keep1 = (contours.loc[:, 'xmax'] - contours.loc[:, 'xmin']) < maxsize
    keep2 = (contours.loc[:, 'ymax'] - contours.loc[:, 'ymin']) < maxsize
    contours = contours.loc[keep1, :]
    contours = contours.loc[keep2, :]

    # Convert anchor df to needed format
    contours = contours.apply(_maybe_coordstr, axis=1)
    contours.loc[:, 'bbox_area'] = (
        contours.loc[:, 'xmax'] - contours.loc[:, 'xmin']
    ) * (contours.loc[:, 'ymax'] - contours.loc[:, 'ymin'])

    # for visualization
    contours.loc[:, 'color'] = contours.loc[:, 'group'].apply(
        lambda x: das.STANDARD_STYLES[x]['lineColor'])

    # Get fovbounds & make everything relative to it
    fovbounds = {
        'XMIN': int(np.min(contours.xmin)),
        'YMIN': int(np.min(contours.ymin)),
        'XMAX': int(np.max(contours.xmax)),
        'YMAX': int(np.max(contours.ymax)),
    }
    fovbounds['appendStr'] = fmeta['appendStr']

    for c in ('x', 'y'):
        # adjust bounding box
        offset = -fovbounds[f'{c.upper()}MIN']
        for bstr in (f'{c}min', f'{c}max'):
            contours.loc[:, bstr] = contours.loc[:, bstr].apply(
                lambda x: int(fmeta['sf'] * (x + offset)))
        # adjust coordinate string
        cstr = f'coords_{c}'
        contours.loc[:, cstr] = contours.loc[:, cstr].apply(
            _adjust_coordstr, offset=offset, sf=fmeta['sf'])

    return contours, fovbounds


def _get_roi_from_contours(
        gc, contours: DataFrame, GTCodes_df: DataFrame,
        fovbounds: dict, fmeta: Union[dict, Series]):
    """"""
    # anchor mask and rgb
    roi_out = {
        'contours': contours.loc[:, [
            'anchor_id', 'group',
            'xmin', 'ymin', 'xmax', 'ymax'
        ]],
        'mask': contours_to_labeled_object_mask(
            contours=contours.copy(), gtcodes=GTCodes_df,
            mode='object'
        ),
        'rgb': _get_rgb_for_interrater(
            gc=gc, bounds=fovbounds, slide_id=fmeta['slide_id']
        ),
    }

    # resize mask to rgb in case there's a couple of pixel
    # difference due to float rounding errors
    roi_out['mask'] = np.uint8(resize(
        roi_out['mask'],
        output_shape=np.array(roi_out['rgb']).shape[:2],
        order=0, preserve_range=True, anti_aliasing=False))

    # visualize
    roi_out['vis'] = _visualize_annotations_on_rgb(
        rgb=roi_out['rgb'],
        contours_list=contours.to_dict(orient='records'),
        linewidth=0.2, x_offset=0, y_offset=0, text=True)

    return roi_out


def parse_anchors_dataset(
        dbcon, gc, savedir: str, whoistruth: str, evalset: str,
        min_side: int = 100):
    """"""
    where = opj(savedir, f'{whoistruth}AreTruth_{evalset}')
    _maybe_mkdir(where)
    _maybe_mkdir(opj(where, 'rgb'))
    _maybe_mkdir(opj(where, 'mask'))
    _maybe_mkdir(opj(where, 'vis'))
    _maybe_mkdir(opj(where, 'contours'))

    # GT codes dict for parsing into label mask
    GTCODE_PATH = '/home/mtageld/Desktop/cTME/ctme/configs/nucleus_GTcodes.csv'
    GTCodes_dict = read_csv(GTCODE_PATH)
    GTCodes_dict.index = GTCodes_dict.loc[:, 'group']
    GTCodes_dict = GTCodes_dict.to_dict(orient='index')
    GTCodes_df = DataFrame.from_dict(GTCodes_dict, orient='index')

    # get the metadata for fovs
    fovmetas = _get_and_maybe_update_fovmeta(dbcon)

    for _, fmeta in fovmetas.iterrows():

        contours, fovbounds = _get_and_fix_contours(
            dbcon, whoistruth=whoistruth, evalset=evalset, fmeta=fmeta)

        # handle edge cases
        if contours.shape[0] < 1:
            continue

        if ((fovbounds['XMAX'] - fovbounds['XMIN']) < min_side) or (
                (fovbounds['YMAX'] - fovbounds['YMIN']) < min_side):
            continue

        # parse rgb, mask, visualization, and contours
        roi_out = _get_roi_from_contours(
            gc=gc, contours=contours, GTCodes_df=GTCodes_df,
            fovbounds=fovbounds, fmeta=fmeta)

        # save output
        roinamestr = \
            f"{fmeta['FOVID']}_{fmeta['slide_name']}" \
            f"_left-{fovbounds['XMIN']}_top-{fovbounds['YMIN']}" \
            f"_bottom-{fovbounds['YMAX']}_right-{fovbounds['XMAX']}"
        print('Saving', roinamestr)
        for imtype in ['mask', 'rgb', 'vis']:
            savename = opj(where, imtype, roinamestr + '.png')
            imwrite(im=roi_out[imtype], uri=savename)
        contours.to_csv(opj(where, 'contours', roinamestr + '.csv'))

# %%===========================================================================


def main():

    DATASETNAME = 'CURATED_v1_2020-03-29_EVAL'

    # where to save stuff
    BASEPATH = "/home/mtageld/Desktop/cTME/results/tcga-nucleus/interrater/"
    savedir = opj(BASEPATH, DATASETNAME, 'i1_anchors', 'DATASET')
    # savedir = opj(BASEPATH, DATASETNAME, 'i1_anchors', 'TMP')
    _maybe_mkdir(savedir)

    # connect to sqlite database -- anchors
    dbcon = _connect_to_anchor_db(opj(savedir, '..', '..'))

    # to get FOV RGBs and visualize cluster medoids etc
    gc = CandygramAPI.connect_to_candygram()

    # Create datasets using different inferred truths
    for whoistruth in ir.CONSENSUS_WHOS:
        for evalset in ['E', 'U-control']:
            parse_anchors_dataset(
                dbcon=dbcon, gc=gc, savedir=savedir,
                whoistruth=whoistruth, evalset=evalset)


# %%===========================================================================

if __name__ == '__main__':
    main()
