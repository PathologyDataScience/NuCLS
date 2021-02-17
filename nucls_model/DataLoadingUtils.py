from collections import Counter
from copy import deepcopy
from imageio import imread
import numpy as np
import os
from os.path import join as opj
from pandas import DataFrame, read_sql_query, read_csv
import matplotlib.pylab as plt
import torch
from PIL import Image
from sklearn.model_selection import KFold

from GeneralUtils import connect_to_sqlite, reverse_dict  # noqa
import nucls_model.torchvision_detection_utils.transforms as tvdt  # noqa
from nucls_model.torchvision_detection_utils.utils import collate_fn  # noqa
from configs.nucleus_style_defaults import Interrater, \
    DefaultAnnotationStyles, NucleusCategories  # noqa
from nucls_model.DataFormattingUtils import from_dense_to_sparse_object_mask  # noqa
from TorchUtils import transform_dlinput  # noqa


ISCUDA = torch.cuda.is_available()
# FDTYPE = torch.float16 if ISCUDA else torch.float32
FDTYPE = torch.float32


# noinspection PyShadowingNames
def _save_slides_for_fold(
        slides_list: list, fold: int, is_training: bool, savepath: str):
    trainstr = 'train' if is_training else 'test'
    slidesdf = DataFrame(slides_list, columns=['slide_name'])
    slidesdf.loc[:, 'hospital'] = [
        sld.split('TCGA-')[1].split('-')[0] for sld in slides_list]
    slidesdf.loc[:, 'type'] = trainstr
    slidesdf.loc[:, 'fold'] = fold
    slidesdf.to_csv(opj(savepath, f'fold_{fold}_{trainstr}.csv'))


# noinspection PyShadowingNames
def _save_cv_splits_by_slide(slides, n_folds, savepath, random_state=0):
    kf = KFold(n_splits=n_folds, random_state=random_state, shuffle=True)
    fold = 0
    for train_idxs, test_idxs in kf.split(slides):
        _save_slides_for_fold(
            slides_list=slides[train_idxs], fold=fold, is_training=True,
            savepath=savepath)
        _save_slides_for_fold(
            slides_list=slides[test_idxs], fold=fold, is_training=False,
            savepath=savepath)
        fold += 1


# noinspection PyShadowingNames
def _get_train_test_slides_given_hospitals(
        slides, big_hospitals, small_hospitals,
        big_hospital_splits, small_hospital_splits):
    train_big_idxs, test_big_idxs = next(big_hospital_splits)
    train_small_idxs, test_small_idxs = next(small_hospital_splits)

    train_hospitals = list(big_hospitals[train_big_idxs]) + list(
        small_hospitals[train_small_idxs])
    test_hospitals = list(big_hospitals[test_big_idxs]) + list(
        small_hospitals[test_small_idxs])

    train_slides = []
    test_slides = []
    for sld in slides:
        hptl = sld.split('TCGA-')[1].split('-')[0]
        if hptl in train_hospitals:
            train_slides.append(sld)
        elif hptl in test_hospitals:
            test_slides.append(sld)

    return train_slides, test_slides


# noinspection PyShadowingNames
def _save_cv_splits_by_hospital(slides, n_folds, savepath, random_state=0):
    # Some hospitals have waaay more slides than others, so to have a balanced
    # split, we make each fold have one big hospital and a few small hospitals
    slides_to_hospitals = {
        sld: sld.split('TCGA-')[1].split('-')[0] for sld in slides}
    hospital_counts = Counter()
    for slide, hospital in slides_to_hospitals.items():
        hospital_counts[hospital] += 1
    big_hospitals = np.array(
        [j[0] for j in hospital_counts.most_common(n_folds)])
    small_hospitals = np.array([
        j for j in hospital_counts.keys() if j not in big_hospitals])

    kf = KFold(n_splits=n_folds, random_state=random_state, shuffle=True)
    big_hospital_splits = kf.split(big_hospitals)
    small_hospital_splits = kf.split(small_hospitals)

    fold = 0
    while fold < n_folds:
        fold += 1

        train_slides, test_slides = _get_train_test_slides_given_hospitals(
            slides=slides, big_hospitals=big_hospitals,
            small_hospitals=small_hospitals,
            big_hospital_splits=big_hospital_splits,
            small_hospital_splits=small_hospital_splits)

        _save_slides_for_fold(
            slides_list=train_slides, fold=fold, is_training=True,
            savepath=savepath)
        _save_slides_for_fold(
            slides_list=test_slides, fold=fold, is_training=False,
            savepath=savepath)


# noinspection PyShadowingNames
def save_cv_train_test_splits(
        root: str, savepath: str, n_folds=5, by_hospital=True, random_state=0):
    slides = np.array(list({
        j.split('_id')[0] for j in sorted(os.listdir(opj(root, 'rgbs')))
        if j.endswith('.png')}))
    if by_hospital:
        _save_cv_splits_by_hospital(
            slides=slides, n_folds=n_folds, savepath=savepath,
            random_state=random_state)
    else:
        _save_cv_splits_by_slide(
            slides=slides, n_folds=n_folds, savepath=savepath,
            random_state=random_state)


# noinspection PyShadowingNames
def get_cv_fold_slides(train_test_splits_path, fold):
    train_slides = read_csv(
            opj(train_test_splits_path, f'fold_{fold}_train.csv')
        ).loc[:, 'slide_name'].to_list()
    test_slides = read_csv(
            opj(train_test_splits_path, f'fold_{fold}_test.csv')
        ).loc[:, 'slide_name'].to_list()
    return train_slides, test_slides


# noinspection PyShadowingNames
def get_transform(**kwargs):
    """Transform data to be passed to model (wrapper)."""
    return transform_dlinput(**kwargs)


# noinspection PyShadowingNames,PyAttributeOutsideInit
class NucleusDatasetBase(object):
    def __init__(
            self, root: str, dbpath: str, slides=None,
            do_classification=False, do_segmentation=False,
            crop_to_fov=False, crop_size=300, crop_plusminus=None,
            transforms=None, load_once=True, bboxes_too=True,
            use_supercategs=False):

        self.root = root
        self.dbpath = dbpath
        self.do_classification = do_classification
        self.do_segmentation = do_segmentation
        self.crop_to_fov = crop_to_fov
        self.crop_size = crop_size
        self.crop_plusminus = None if crop_to_fov else crop_plusminus
        self.transforms = get_transform() if transforms is None else transforms
        self.load_once = load_once
        self.use_supercategs = use_supercategs

        # contiguous label code map for classification
        self.set_labelmaps()

        # same but for supercategs (whether or not they are used)
        self.set_supercateg_labelmaps()

        # Note: the total number of nuclei in the QC'd *training* set
        #  is reasonable to load in memory to prevent back and forth
        #  communication with disk. Also note that sqlite interferes with
        #  pytorch multithreading, so using num_workers > 1 when
        #  loading data requires self.load_once to be True
        dbcon = connect_to_sqlite(self.dbpath)
        self.dbcon = dbcon if not load_once else None

        # Note: the crop is IMPORTANT to ensure the size of images
        # fed to the network during training is constant. It's not just for
        # augmentation!
        self.cropper = tvdt.Cropper(
            size=self.crop_size, plusminus=self.crop_plusminus)

        # set names of fovs, slides, load boxes, etc
        self.set_fovnames_and_ids(dbcon=dbcon, slides=slides)
        self.maybe_load_fovlocs_and_boxes(dbcon=dbcon, bboxes_too=bboxes_too)

        # assign a weight to each fov to emphasize fovs with 1. uncommon
        # nuclei categories, 2. more nuclei
        self.set_fovweights(dbcon=dbcon)

    def set_labelmaps(self):
        ncg = NucleusCategories
        if not self.do_classification:
            self.categs_names = ncg.puredet_categs
            self.labelcodes = ncg.raw_to_puredet_categs_codes
            self.rlabelcodes = reverse_dict(ncg.puredet_categs_codes)
        elif self.use_supercategs:
            self.categs_names = ncg.super_categs
            self.labelcodes = ncg.raw_to_super_categs_codes
            self.rlabelcodes = reverse_dict(ncg.super_categs_codes)
        else:
            self.categs_names = ncg.main_categs
            self.labelcodes = ncg.raw_to_main_categs_codes
            self.rlabelcodes = reverse_dict(ncg.main_categs_codes)

    def set_supercateg_labelmaps(self):
        ncg = NucleusCategories
        if self.do_classification:
            self.supercategs_names = ncg.super_categs
            self.supercategs_rlabelcodes = reverse_dict(ncg.super_categs_codes)
            self.main_codes_to_supercategs_names = {
                ncg.main_categs_codes[mc]: sc
                for mc, sc in ncg.main_to_super_categmap.items()
            }
            self.main_codes_to_supercategs_codes = {
                k: ncg.super_categs_codes[v]
                for k, v in self.main_codes_to_supercategs_names.items()
            }

    def set_fovnames_and_ids(self, dbcon, slides: list):
        # get fovlist and connect to db
        self.fovnames = [
            j.split('.png')[0] for j in
            sorted(os.listdir(opj(self.root, 'rgbs'))) if j.endswith('.png')
        ]
        # connect fovnames to fovids in db for efficiency
        fovids = read_sql_query(f"""
            SELECT fovname, fov_id, slide_name FROM fov_meta
            WHERE slide_name LIKE "TCGA-%-DX1%"
        """, con=dbcon)
        sl = fovids.loc[:, 'slide_name']
        fovids.loc[:, 'slide_name'] = sl.apply(lambda s: s.split('_id')[0])
        fovids.index = fovids.loc[:, 'slide_name']
        if slides is not None:
            slds = set(fovids.index)
            fovids = fovids.loc[slds.intersection(slides), :]
        self.slides = list(set(fovids.index))
        fovids.index = fovids.loc[:, 'fovname']
        self.fovnames = [j for j in self.fovnames if j in fovids.index]
        if len(self.fovnames) < 1:
            raise ValueError("There are no FOVs for the requested slides!!")
        self.fovids = fovids.loc[self.fovnames, 'fov_id'].to_dict()
        self.rfovids = reverse_dict(self.fovids)

    def maybe_load_fovlocs_and_boxes(self, dbcon, bboxes_too: bool):
        if self.load_once:
            # read all fov locations
            fovstr = Interrater._get_sqlitestr_for_list(self.rfovids.keys())
            self.fovloc = read_sql_query(f"""
                SELECT "fov_id", "xmin", "ymin", "xmax", "ymax"
                FROM "annotation_elements"
                WHERE "fov_id" IN ({fovstr})
                  AND "group" LIKE "fov%"
            ;""", con=dbcon)

            # load all bounding boxes to avoid sqlite need during training
            if bboxes_too:
                fovstr = Interrater._get_sqlitestr_for_list(
                    self.rfovids.keys())
                self.boxdf = read_sql_query(f"""
                    SELECT "fov_id", "xmin", "ymin", "xmax", "ymax", 
                           "group", "type"
                    FROM "annotation_elements"
                    WHERE "fov_id" IN ({fovstr})
                      AND "group" NOT LIKE "fov%"
                ;""", con=dbcon)

    def set_fovweights(self, dbcon):
        """
        Fov weight is the WEIGHTED AVERAGE weight of nuclei in it, with
        less abundant nuclei receiving higher weight, and fovs with more
        nuclei DENSITY (i.e. more informative) receiving higher weight. We
        use density (no per pixel) since the train loader crops a constant
        size region from each fov. These weights are used by the RandomSampler
        to mitigate class imbalance.
        """
        fovstr = Interrater._get_sqlitestr_for_list(self.rfovids.keys())

        # raw overall frequency of nucleus categs
        totcounts_df = read_sql_query(f"""
            SELECT "group", count(*) AS "no"
            FROM "annotation_elements"
            WHERE "fov_id" IN ({fovstr})
              AND "group" NOT LIKE "fov%"
            GROUP BY "group"
        ;""", con=dbcon)

        # group as needed and get total counts
        totcounts_df.loc[:, 'group'] = totcounts_df.loc[:, 'group'].map(
            self.labelcodes)
        self.categcounts = self._get_categcounts_from_df(totcounts_df)

        # Assign weights. Weights are inversely proportional to the relative
        # frequency of class, with the exception of ambiguous classes which
        # are assigned zero weight
        maxval = self.categcounts.most_common(1)[0][1]
        self.categweights = {
            k: maxval / v for k, v in self.categcounts.items()}
        self.categweights[len(self.categs_names)] = 1e-5  # ignore ambiguous
        totwt = sum(self.categweights.values())
        self.categweights = Counter({
            k: v / totwt for k, v in self.categweights.items()})

        # frequency of categs for each fov
        cntperfov_df = read_sql_query(f"""
            SELECT "fov_id", "group", count(*) AS "no"
            FROM "annotation_elements"
            WHERE "fov_id" IN ({fovstr})
              AND "group" NOT LIKE "fov%"
            GROUP BY "fov_id", "group"
        ;""", con=dbcon)
        cntperfov_df.loc[:, 'group'] = cntperfov_df.loc[:, 'group'].map(
            self.labelcodes)

        # IMPORTANT NOTE: It is critical that the weights are in the same
        #  order as self.fovnames, since the self.getitem() method uses the
        #  index relative to self.fovnames
        self.fov_weights = []
        for fovname in self.fovnames:

            fovid = self.fovids[fovname]

            # get total counts for this fov
            fov_categcounts = self._get_categcounts_from_df(
                cntperfov_df.loc[cntperfov_df.loc[:, 'fov_id'] == fovid, :])
            # fov area
            coords = self.fovloc.loc[
                self.fovloc.loc[:, 'fov_id'] == fovid, :].iloc[0, :]
            area = (coords['xmax'] - coords['xmin']) * (
                coords['ymax'] - coords['ymin'])
            # unnormalized weight
            self.fov_weights.append(sum([
                self.categweights[cat] * cnt / area
                for cat, cnt in fov_categcounts.items()
            ]))

        # normalize so everything sums to 1
        totwt = sum(self.fov_weights)
        self.fov_weights = [j / totwt for j in self.fov_weights]

    @staticmethod
    def _get_categcounts_from_df(counts_df):
        categcounts = Counter()
        for _, cat in counts_df.iterrows():
            if cat['group'] in categcounts:
                categcounts[cat['group']] += cat['no']
            else:
                categcounts[cat['group']] = cat['no']
        return categcounts

    def __len__(self):
        return len(self.fovnames)


# noinspection PyShadowingNames
class NucleusDataset(NucleusDatasetBase):
    def __init__(
            self, root: str, dbpath: str, slides=None, do_classification=False,
            crop_to_fov=False, crop_size=256, crop_plusminus=None,
            transforms=None, load_once=True, use_supercategs=False):

        super(NucleusDataset, self).__init__(
            root=root, dbpath=dbpath, slides=slides,
            do_classification=do_classification,
            do_segmentation=False,  # This is ONLY for FasterRCNN
            crop_to_fov=crop_to_fov, crop_size=crop_size,
            crop_plusminus=crop_plusminus, transforms=transforms,
            load_once=load_once, bboxes_too=True,
            use_supercategs=use_supercategs,
        )

    def __getitem__(self, idx):
        return self.getitem(idx)

    def _get_boxdf(self, fovname):
        if self.load_once:
            boxdf = self.boxdf.loc[self.boxdf.loc[
                :, 'fov_id'] == self.fovids[fovname], :].copy()
        else:
            boxdf = read_sql_query(f"""
                SELECT "xmin", "ymin", "xmax", "ymax", "group", "type"
                FROM "annotation_elements"
                WHERE "fov_id" = {self.fovids[fovname]}
                  AND "group" NOT LIKE "fov%"
            ;""", con=self.dbcon)
        return boxdf

    def _get_fovloc(self, fovname):
        if self.load_once:
            fovloc = self.fovloc.loc[
                self.fovloc.loc[:, 'fov_id'] == self.fovids[fovname], :].copy()
        else:
            fovloc = read_sql_query(f"""
                SELECT "fov_id", "xmin", "ymin", "xmax", "ymax"
                FROM "annotation_elements"
                WHERE "fov_id" = {self.fovids[fovname]}
                  AND "group" LIKE "fov%"
            ;""", con=self.dbcon)
        return fovloc.iloc[0, 1:]

    def getitem(self, idx, target=None, boxdf=None):

        target = {} if target is None else target

        fovname = self.fovnames[idx]

        # load image
        # IMPORTANT NOTE: we're keeping the RGB as a PIL image
        # because this is required for cropping and other transforms
        # rgb_path = opj(self.root, "rgbs", f"{fovname}.png")
        rgb_path = opj(self.root, "rgbs_colorNormalized", f"{fovname}.png")
        rgb = Image.fromarray(imread(rgb_path)[..., :3])

        # get bounding box coordinates for each object
        boxdf = self._get_boxdf(fovname=fovname) if boxdf is None else boxdf
        w, h = rgb.size
        _, keep = tvdt.remove_degenerate_bboxes(
            boxes=boxdf.loc[:, ['xmin', 'ymin', 'xmax', 'ymax']].values,
            dim0=w, dim1=h, min_boxside=self.cropper.min_boxside)
        boxdf = boxdf.iloc[keep, :]
        boxes = torch.as_tensor(
            boxdf.loc[:, ['xmin', 'ymin', 'xmax', 'ymax']].values,
            dtype=FDTYPE)

        if 'ismask' in target:
            target['ismask'] = target['ismask'][keep]

        # Get fov location. Note that in the dataset, the FOV used for
        # annotation is slighly smaller than the RGB in the final dataset
        # to allow for good handling of nuclei at edge
        fovloc_array = self._get_fovloc(fovname=fovname)
        fovloc = fovloc_array.to_dict()

        # update target
        target.update({
            'boxes': boxes,
            'image_id': torch.tensor([self.fovids[fovname]]),
            'n_objects': torch.tensor([boxes.shape[0]]),
            'area': (boxes[:, 3] - boxes[:, 1]) * (
                    boxes[:, 2] - boxes[:, 0]),
            'fovloc': torch.tensor(fovloc_array.values, dtype=torch.int32),
        })

        # map to standard classes (which may be pure detection/ambiguous)
        labels = boxdf.loc[:, 'group'].map(self.labelcodes).values
        target['labels'] = torch.as_tensor(labels, dtype=torch.int64)

        # nuclei that should be ignored in evaluation (but used in training)
        # this is used by the cocoEvaluator, which ignore the GROUND TRUTH
        # nuclei tagged as iscrowd .. I also modified the model itself to
        # discard predictions with the ambiguous category in evaluation mode
        target['iscrowd'] = torch.as_tensor(
            labels == self.labelcodes[
                NucleusCategories.ambiguous_categs[0]], dtype=FDTYPE)

        # maybe crop to fov
        if self.crop_to_fov:
            rgb, target = self.cropper(
                rgb=rgb, targets=target,
                i=fovloc['ymin'], h=fovloc['ymax'] - fovloc['ymin'],
                j=fovloc['xmin'], w=fovloc['xmax'] - fovloc['xmin'],
            )

        # maybe random crop (possibly after cropping to fov)
        if self.crop_size is not None:
            rgb, target = self.cropper(rgb=rgb, targets=target)

        # apply transforms (eg random flip)
        rgb, target = self.transforms(rgb, target)

        return rgb, target


# noinspection PyShadowingNames
class NucleusDatasetMask(NucleusDataset):
    def __init__(
            self, root: str, dbpath: str, slides=None, do_classification=False,
            crop_to_fov=False, crop_size=256, crop_plusminus=None,
            transforms=None, load_once=True, use_supercategs=False):

        super(NucleusDatasetMask, self).__init__(
            root=root, dbpath=dbpath, slides=slides,
            do_classification=do_classification,
            crop_to_fov=crop_to_fov, crop_size=crop_size,
            crop_plusminus=crop_plusminus, transforms=transforms,
            load_once=load_once, use_supercategs=use_supercategs,
        )

        self.do_segmentation = True  # by definition
        self.fovcode = DefaultAnnotationStyles.gtcodes_dict['fov_basic']['GT_code']  # noqa

    def __getitem__(self, idx):

        fovname = self.fovnames[idx]

        # get nuclei bounding boxes
        boxdf = self._get_boxdf(fovname=fovname)

        # load mask
        mask = imread(opj(self.root, "mask", f"{fovname}.png"))[..., :3]

        # get rid of FOV annotation
        fovmask = mask[..., 0] == self.fovcode
        mask[fovmask] = 0

        # init & assign target
        target = {
            'dense_mask': Image.fromarray(mask),

            # tag which nuclei are segmentations and which are bboxes
            'ismask': torch.as_tensor(
                boxdf.loc[:, 'type'].values == 'polyline',
                dtype=FDTYPE),
        }

        # mTODO (?): find total overlap of each segmented nucleus with any
        #  surroundings. Note that if a bbox lies "in front" of a segment.
        #  this would obscure part of segmentation. The nuclei should
        #  be given low weight in training and ignored in evaluation.
        #  Also note that this is an artifact caused by this function:
        #    histomicstk.annotations_and_masks.\
        #    annotations_to_object_mask_handler.\
        #    contours_to_labeled_object_mask
        #  which does not handle "layering" (i.e. order of overlay)
        #  of individual annotation elements. It partially handles this
        #  though by overlaying the smallest nuclei on top, making this
        #  issue mostly inconsequential.

        # pass image and target through cropper, transforms etc
        rgb, target = self.getitem(idx, target=target, boxdf=boxdf)

        # split the flat mask into a set of binary masks
        masks, _ = from_dense_to_sparse_object_mask(
            dense_mask=np.uint8(target['dense_mask']),
            boxes=np.int32(target['boxes']))

        # convert to tensor
        del target['dense_mask']
        target['masks'] = torch.as_tensor(masks, dtype=torch.uint8)

        return rgb, target


# noinspection PyShadowingNames,PyPep8Naming
class NucleusDatasetMask_IMPRECISE(NucleusDataset):
    """This is an IMPRECISE implementation because it extracts
    the nucleus label and whether or not it's a bounding box
    from the mask itself, so it causes two problems:

    1. The labels will be imprecise as the center of the bounding box
       may actually be occupied by another overlapping bbox
    2. Any overlapping bounding box is considered to be a mask.

    DO NOT USE THIS IF POSSIBLE!!!
    """
    def __init__(
            self, root: str, dbpath: str, slides=None, do_classification=False,
            crop_to_fov=False, crop_size=256, crop_plusminus=None,
            transforms=None, load_once=True, use_supercategs=False):

        super(NucleusDatasetMask_IMPRECISE, self).__init__(
            root=root, dbpath=dbpath, slides=slides,
            do_classification=do_classification,
            crop_to_fov=crop_to_fov, crop_size=crop_size,
            crop_plusminus=crop_plusminus, transforms=transforms,
            load_once=load_once, use_supercategs=use_supercategs,
        )

        self.do_segmentation = True  # by definition
        self.fovcode = DefaultAnnotationStyles.gtcodes_dict['fov_basic']['GT_code']  # noqa

    def __getitem__(self, idx):

        fovname = self.fovnames[idx]

        # load image
        # IMPORTANT NOTE: we're keeping the RGB as a PIL image
        # because this is required for cropping and other transforms
        # rgb_path = os.path.join(self.root, "rgbs", f"{fovname}.png")
        rgb_path = opj(self.root, "rgbs_colorNormalized", f"{fovname}.png")
        rgb = Image.fromarray(imread(rgb_path)[..., :3])

        # load mask
        # loaded_mask = Image.fromarray(
        #     imread(rgb_path.replace('rgbs', 'mask'))[..., :3])
        loaded_mask = Image.fromarray(
            imread(rgb_path.replace('rgbs_colorNormalized', 'mask'))[..., :3])
        target = {'dense_mask': loaded_mask}

        # maybe crop to fov
        if self.crop_to_fov:
            if self.load_once:
                fovloc = self.fovloc.loc[self.fovloc.loc[
                    :, 'fov_id'] == self.fovids[fovname], :].copy()
            else:
                fovloc = read_sql_query(f"""
                    SELECT "xmin", "ymin", "xmax", "ymax"
                    FROM "annotation_elements"
                    WHERE "fov_id" = {self.fovids[fovname]}
                      AND "group" LIKE "fov%"
                ;""", con=self.dbcon)
            fovloc = fovloc.iloc[0, :].to_dict()
            rgb, target = self.cropper(
                rgb=rgb, targets=target,
                i=fovloc['ymin'], w=fovloc['ymax'] - fovloc['ymin'],
                j=fovloc['xmin'], h=fovloc['xmax'] - fovloc['xmin'],
            )

        # maybe random crop (possibly after cropping to fov)
        if self.crop_size is not None:
            rgb, target = self.cropper(rgb=rgb, targets=target)

        # apply transforms (eg random flip)
        rgb, target = self.transforms(rgb, target)

        # The fov location is not an object
        loaded_mask = deepcopy(np.uint8(target['dense_mask']))
        fovcode = self.fovcode
        loaded_mask[loaded_mask[..., 0] == fovcode, :] = 0

        # split the color-encoded mask into a set of binary masks
        masks, _ = from_dense_to_sparse_object_mask(dense_mask=loaded_mask)

        # get bounding box coordinates for each mask
        ncg = NucleusCategories
        num_objs = masks.shape[0]
        boxes = []
        ismask = []
        areas = []
        labels = []
        for i in range(num_objs):
            pos = np.where(masks[i, ...])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])

            # FIXME: the following assumes that the nucleus pixels are the
            #  most common pixels within the bounding box
            unique, count = np.unique(
                loaded_mask[ymin:ymax, xmin:xmax, 0], return_counts=True)
            if (len(unique) > 0) and (unique[0] > 0):
                raw = ncg.rgtcodes_dict[unique[np.argmax(count)]]['group']
            else:
                continue

            # Handle if centre of bbox falls outside any annotation
            if raw not in ncg.raw_categs:
                continue

            boxes.append([xmin, ymin, xmax, ymax])
            areas.append((xmax - xmin) * (ymax - ymin))

            # tag which nuclei are segmentations and which are bboxes
            ix = len(boxes) - 1
            ismask.append(np.sum(masks[i, ...]) < areas[ix])

            # handle classification
            if self.do_classification:
                if not self.use_supercategs:
                    lbl = ncg.raw_to_main_categs_codes[raw]
                else:
                    lbl = ncg.raw_to_super_categs_codes[raw]
                labels.append(lbl)
            else:
                labels.append(1)

        # pack into target dict
        target.update({
            'boxes': torch.as_tensor(boxes, dtype=FDTYPE),
            'image_id': torch.tensor([self.fovids[fovname]]),
            'n_objects': torch.tensor([num_objs]),
            'area': torch.as_tensor(areas, dtype=FDTYPE),

            # mask vs bbox
            'ismask': torch.tensor(ismask),

            # # ignore this nucleus during evaluation?
            # 'iscrowd': torch.tensor([False] * len(boxes)),
            # nuclei that should be ignored in evaluation (but used in training)
            # this is used by the cocoEvaluator, which ignore the GROUND TRUTH
            # nuclei tagged as iscrowd .. I also modified the model itself to
            # discard predictions with the ambiguous category in evaluation mode
            'iscrowd': torch.as_tensor(
                labels == self.labelcodes[
                    NucleusCategories.ambiguous_categs[0]], dtype=FDTYPE
            ),

            'labels': torch.as_tensor(labels, dtype=torch.int64),
        })

        # convert to tensor
        del target['dense_mask']
        target['masks'] = torch.as_tensor(masks, dtype=torch.uint8)

        return rgb, target


# noinspection PyShadowingNames
def _crop_all_to_fov(
        images, targets, outputs=None, cropper=None, crop_targets=True):
    """Crop to fov so that the model looks at a wider field than it
    does inference? This is also important since the dataset I made
    deliberately looks beyond FOV to include full extent of nuclei that
    spill over the FOV edge.
    """
    imgs = deepcopy(images)
    trgs = deepcopy(targets)
    outs = [None] * len(imgs) if outputs is None else deepcopy(outputs)
    cropper = tvdt.Cropper() if cropper is None else cropper
    for imidx, (img, trg, out) in enumerate(zip(imgs, trgs, outs)):
        img, _ = tvdt.ToPILImage()(img.cpu(), {})
        xmin, ymin, xmax, ymax = [int(j) for j in trg['fovloc']]
        rgb1 = None
        rgb2 = None
        if crop_targets:
            rgb1, trg = cropper(
                rgb=img, targets=trg,
                i=ymin, h=ymax - ymin, j=xmin, w=xmax - xmin,
            )
        if out is not None:
            out = {k: v.detach() for k, v in out.items() if torch.is_tensor(v)}
            rgb2, out = cropper(
                rgb=img, targets=out,
                i=ymin, h=ymax - ymin, j=xmin, w=xmax - xmin,
            )
        # now assign
        rgb = rgb1 or rgb2
        imgs[imidx], _ = tvdt.PILToTensor()(rgb, {})
        trgs[imidx] = trg
        outs[imidx] = out

    return imgs, trgs, outs


# %% ==========================================================================

