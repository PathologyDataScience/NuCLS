# -*- coding: utf-8 -*-
"""
Created on Fri Jul 27 16:37:21 2018

@author: tageldim
"""
import sys
from pandas import read_sql_query, DataFrame, read_csv
from typing import Union
from collections import OrderedDict

import os
from os.path import join as opj
BASEPATH = opj(os.path.expanduser('~'), 'Desktop', 'NuCLS')
sys.path.insert(0, BASEPATH)
from ctme.GeneralUtils import ordered_vals_from_ordered_dict  # noqa

# %%===========================================================================
# Default nucleus styles


class DefaultAnnotationStyles(object):

    DEFAULT_LINEWIDTH = 3.0
    DEFAULT_OPACITY = 0

    # FOV styles
    FOV_STYLES = {
        "fov_preapproved": {"lineColor": "rgb(255,10,255)"},
        "fov_for_programmatic_edit": {"lineColor": "rgb(255,0,0)"},
        "fov_basic": {"lineColor": "rgb(0,0,0)"},
        "fov_representative": {"lineColor": "rgb(255,255,255)"},
        "fov_problematic": {"lineColor": "rgb(255,255,0)"},
        "fov_discordant": {"lineColor": "rgb(0,0,0)"},
    }
    for k in FOV_STYLES.keys():
        FOV_STYLES[k]['type'] = 'rectangle'

    # *** Standard styles for final dataset ***
    STANDARD_STYLES = {
        "tumor": {"lineColor": "rgb(255,0,0)"},
        "mitotic_figure": {"lineColor": "rgb(255,191,0)"},
        "fibroblast": {"lineColor": "rgb(0,230,77)"},
        "lymphocyte": {"lineColor": "rgb(0,0,255)"},
        "plasma_cell": {"lineColor": "rgb(0,255,255)"},
        "macrophage": {"lineColor": "rgb(51,102,153)"},
        "neutrophil": {"lineColor": "rgb(51,102,204)"},
        "eosinophil": {"lineColor": "rgb(128,0,0)"},
        "apoptotic_body": {"lineColor": "rgb(255,128,0)"},
        "vascular_endothelium": {"lineColor": "rgb(102,0,204)"},
        "myoepithelium": {"lineColor": "rgb(250,50,250)"},
        "ductal_epithelium": {"lineColor": "rgb(255,128,255)"},
        "unlabeled": {"lineColor": "rgb(80,80,80)"},
    }
    for k in STANDARD_STYLES.keys():
        STANDARD_STYLES[k]['type'] = "polyline"

    # standard class names and colors in plt-compatible format
    MAIN_CLASSES = ['tumor', 'fibroblast', 'lymphocyte']
    CLASSES = list(STANDARD_STYLES.keys())
    COLORS = {
        c: [
            int(v) / 255 for v in
            s['lineColor'].split('rgb(')[1].split(')')[0].split(',')
        ] for c, s in STANDARD_STYLES.items()
    }
    COLORS['all'] = COLORS['unlabeled']
    COLORS['detection'] = COLORS['unlabeled']
    COLORS['classification'] = COLORS['unlabeled']

    STANDARD_STYLES.update(FOV_STYLES)  # standard styles incl. fovs

    # assign other default annotation document attributes
    for stylename in STANDARD_STYLES.keys():

        style = STANDARD_STYLES[stylename]

        # by default, group and label are assigned dict key
        STANDARD_STYLES[stylename]["group"] = stylename
        STANDARD_STYLES[stylename]["label"] = {"value": stylename}

        # common defaults
        STANDARD_STYLES[stylename]["lineWidth"] = DEFAULT_LINEWIDTH
        fillColor = style['lineColor']
        fillColor = fillColor.replace("rgb", "rgba")
        fillColor = fillColor[:fillColor.rfind(")")] + ",{})".format(
            DEFAULT_OPACITY)

        # type-specific defaults
        if STANDARD_STYLES[stylename]["type"] == "rectangle":
            STANDARD_STYLES[stylename]['center'] = [0, 0, 0]
            STANDARD_STYLES[stylename]['height'] = 1
            STANDARD_STYLES[stylename]['width'] = 1
            STANDARD_STYLES[stylename]['rotation'] = 0
            STANDARD_STYLES[stylename]['normal'] = [0, 0, 1]
            STANDARD_STYLES[stylename]["lineWidth"] = 6

        elif STANDARD_STYLES[stylename]["type"] == "polyline":
            STANDARD_STYLES[stylename]['closed'] = True
            STANDARD_STYLES[stylename]['points'] = [
                [0, 0, 0], [0, 1, 0], [1, 0, 0]]
            STANDARD_STYLES[stylename]["fillColor"] = fillColor

    # GT codes dict for parsing into label mask
    GTCODE_PATH = opj(BASEPATH, 'ctme/configs/nucleus_GTcodes.csv')
    gtcodes_df = read_csv(GTCODE_PATH)
    gtcodes_df.index = gtcodes_df.loc[:, 'group']
    gtcodes_dict = gtcodes_df.to_dict(orient='index')

    # reverse dict for quick access (indexed by GTcode in mask)
    # NOTE: since multiple fov styles share the same GTcode, they map to
    # the same key as 'fov_discordant'
    rgtcodes_dict = {v['GT_code']: v for k, v in gtcodes_dict.items()}

# %%===========================================================================
# Nucleus categories for training models


class NucleusCategories(object):

    das = DefaultAnnotationStyles

    # ground truth codes as they appear in masks
    gtcodes_dict = das.gtcodes_dict
    rgtcodes_dict = das.rgtcodes_dict

    # categories that should only be used to TRAIN detector and which have
    # no classification because of their ambiguity and high discordance
    ambiguous_categs = [
        'apoptotic_body',
        'unlabeled',
    ]
    ambiguous_categs.extend(
        [f'correction_{v}' for v in ambiguous_categs]
    )

    # map from raw categories to main categories to be learned
    raw_to_main_categmap = OrderedDict({
        'tumor': 'tumor_nonMitotic',
        'mitotic_figure': 'tumor_mitotic',
        'fibroblast': 'nonTILnonMQ_stromal',
        'vascular_endothelium': 'nonTILnonMQ_stromal',
        'macrophage': 'macrophage',
        'lymphocyte': 'lymphocyte',
        'plasma_cell': 'plasma_cell',
        'neutrophil': 'other_nucleus',
        'eosinophil': 'other_nucleus',
        'myoepithelium': 'other_nucleus',
        'ductal_epithelium': 'other_nucleus',
    })
    raw_to_main_categmap.update({
        f'correction_{k}': v for k, v in raw_to_main_categmap.items()
    })
    raw_to_main_categmap.update({k: 'AMBIGUOUS' for k in ambiguous_categs})
    raw_categs = raw_to_main_categmap.keys()

    # map from main categories to super-categories
    main_to_super_categmap = OrderedDict({
        'tumor_nonMitotic': 'tumor_any',
        'tumor_mitotic': 'tumor_any',
        'nonTILnonMQ_stromal': 'nonTIL_stromal',
        'macrophage': 'nonTIL_stromal',
        'lymphocyte': 'sTIL',
        'plasma_cell': 'sTIL',
        'other_nucleus': 'other_nucleus',
        'AMBIGUOUS': 'AMBIGUOUS',
    })

    # same but from main to supercategs
    raw_to_super_categmap = OrderedDict()
    for k, v in raw_to_main_categmap.items():
        raw_to_super_categmap[k] = main_to_super_categmap[v]

    # names & *contiguous* gt codes for main categories
    main_categs = ordered_vals_from_ordered_dict(raw_to_main_categmap)
    main_categs_codes = {j: i + 1 for i, j in enumerate(main_categs)}

    # names & *contiguous* gt codes for super categories
    super_categs = ordered_vals_from_ordered_dict(main_to_super_categmap)
    super_categs_codes = {j: i + 1 for i, j in enumerate(super_categs)}

    # direct dict mapping from main categs to super category gt codes
    main_codes_to_super_codes = {}
    for k, v in main_categs_codes.items():
        main_codes_to_super_codes[v] = super_categs_codes[
            main_to_super_categmap[k]]

    # direct dict mapping from raw categs to main category gt codes
    raw_to_main_categs_codes = {}
    for k, v in raw_to_main_categmap.items():
        raw_to_main_categs_codes[k] = main_categs_codes[v]

    # direct dict mapping from raw categs to super category gt codes
    raw_to_super_categs_codes = {}
    for k, v in raw_to_super_categmap.items():
        raw_to_super_categs_codes[k] = super_categs_codes[v]

    # map from raw categories to PURE detection categories
    raw_to_puredet_categmap = OrderedDict()
    for k in raw_to_main_categmap.keys():
        raw_to_puredet_categmap[k] = \
            'nucleus' if k not in ambiguous_categs else 'AMBIGUOUS'

    # names & *contiguous* gt codes for PURE detection
    puredet_categs = ordered_vals_from_ordered_dict(raw_to_puredet_categmap)
    puredet_categs_codes = {j: i + 1 for i, j in enumerate(puredet_categs)}
    raw_to_puredet_categs_codes = {}
    for k, v in raw_to_puredet_categmap.items():
        raw_to_puredet_categs_codes[k] = puredet_categs_codes[v]

    # categmap from ORIGINAL codes (what the participants SAW)
    # we do this at the level of super-classes because there's no fine
    # distinction at the crude level of data shown
    original_to_super_categmap = OrderedDict({
        # tumor
        'tumor': 'tumor_any',
        # stromal
        'fibroblast': 'nonTIL_stromal',
        'vascular_endothelium': 'nonTIL_stromal',
        # tils
        'lymphocyte': 'sTIL',
        'plasma_cell': 'sTIL',
        'other_inflammatory': 'sTIL',
        # other
        'normal_acinus_or_duct': 'other_nucleus',
        'nerve': 'other_nucleus',
        'skin_adnexa': 'other_nucleus',
        'other': 'other_nucleus',
        'adipocyte': 'other_nucleus',
        # meh / junk
        'necrosis_or_debris': 'AMBIGUOUS',
        'glandular_secretions': 'AMBIGUOUS',
        'blood': 'AMBIGUOUS',
        'exclude': 'AMBIGUOUS',
        'metaplasia_NOS': 'AMBIGUOUS',
        'mucoid_material': 'AMBIGUOUS',
        'lymphatics': 'AMBIGUOUS',
        'undetermined': 'AMBIGUOUS',
    })

    # categmap from REGION codes (what participants 'consulted' while
    # making their assessment) and super-class code.
    regions_to_super_categmap = OrderedDict({
        # tumor
        'tumor': 'tumor_any',
        'angioinvasion': 'tumor_any',
        'dcis': 'tumor_any',
        # stromal
        'stroma': 'nonTIL_stromal',
        'blood_vessel': 'nonTIL_stromal',
        # tils
        'lymphocytic_infiltrate': 'sTIL',
        'plasma_cells': 'sTIL',
        'other_immune_infiltrate': 'sTIL',
        # other
        'normal_acinus_or_duct': 'other_nucleus',
        'nerve': 'other_nucleus',
        'skin_adnexa': 'other_nucleus',
        'other': 'other_nucleus',
        'fat': 'other_nucleus',
        # meh / junk
        'necrosis_or_debris': 'AMBIGUOUS',
        'glandular_secretions': 'AMBIGUOUS',
        'blood': 'AMBIGUOUS',
        'exclude': 'AMBIGUOUS',
        'metaplasia_NOS': 'AMBIGUOUS',
        'mucoid_material':  'AMBIGUOUS',
        'lymphatics': 'AMBIGUOUS',
        'undetermined': 'AMBIGUOUS',
    })
    regions_to_super_categs_codes = {}
    for k, v in regions_to_super_categmap.items():
        regions_to_super_categs_codes[k] = super_categs_codes[v]

# %%===========================================================================
# Naming conventions from nucleus database


class NameStandardization(object):

    # certain things need to be mapped using synonyms as opposed to substrings
    SYNONYMS = {
        'fov': 'fov_basic',
    }

    # map to unified vocbulary -> to train models
    # these are the substrings that map to standardizes styles
    # everythin is also a substring of itself
    CLASSNAME_VOCAB = {
        k: [k] for k in DefaultAnnotationStyles.STANDARD_STYLES.keys()}
    EXTRA_SUBSTRINGS = {
        'fov_preapproved': ['correction_fov'],
        'fov_for_programmatic_edit': [
            'fov_for_programmatic_correction', 'fov_needs_correction'],
        'tumor': ['tumour'],
        'fibroblast': ['fibrobl', 'fiborbl', 'fiobroblast', 'stroma'],
        'lymphocyte': ['lymphocyt'],
        'plasma_cell': ['plasma'],
        'macrophage': ['macrophag', 'histiocyt'],
        'mitotic_figure': ['mitot', 'mitos'],
        'vascular_endothelium': ['endoth', 'vascul'],
        'myoepithelium': ['myoepith', 'myoeputh'],
        'apoptotic_body': ['apopt'],
        'neutrophil': ['neutroph'],
        'ductal_epithelium': ['ductal_epith', 'ductal epith'],
        'eosinophil': ['eosinoph', 'esinoph'],
        'unlabeled': [
            'unlabel', 'mohamed_nucleus', 'debri', 'rbc', 'unknown'
        ],
    }
    for k in CLASSNAME_VOCAB.keys():
        if k in EXTRA_SUBSTRINGS.keys():
            CLASSNAME_VOCAB[k].extend(EXTRA_SUBSTRINGS[k])

    # reverse so that key is the substring
    CLASSNAME_VOCAB_REVERSED = {}
    for k, vlist in CLASSNAME_VOCAB.items():
        for v in vlist:
            CLASSNAME_VOCAB_REVERSED[v] = k

    # irrelevant annotations to ignore
    IGNORE_VOCAB = [
        'raghav_fov', 'roi', 'meh', 'nucleus',
        'slide_complete', 'raghav_nucleus', 'default',
        'blabla', 'correction_lasso_fibroblast',
    ]
    IGNORE_VOCAB = [j.lower() for j in IGNORE_VOCAB]

# %%===========================================================================


class GalleryStyles(object):

    APPROVED = {
        "GOOD": {  # core set (approved by SP)
            "lineColor": "rgb(0,255,0)",
            "fillColor": "rgba(0,255,0,0.3)",
        },
        "CORRECTED": {  # corrected following SP instructions
            "lineColor": "rgb(0,255,100)",
            "fillColor": "rgba(0,255,100,0.3)",
        },
        "COMPLETE": {  # eval sets (by Mohamed)
            "lineColor": "rgb(0,255,0)",
            "fillColor": "rgba(0,255,0,0.3)",
        },
    }
    for k in APPROVED:
        APPROVED[k]['group'] = k

    NOT_APPROVED = {
        "NEEDS_WORK": {  # (as stated by SP)
            "lineColor": "rgb(255,255,0)",
            "fillColor": "rgba(255,255,0,0.3)",
        },
        "BAD": {  # (as stated by SP)
            "lineColor": "rgb(255,0,0)",
            "fillColor": "rgba(255,0,0,0.3)",
        },
        "PREAPPROVED": {  # needs blessing by pathologist
            "lineColor": "rgb(0,0,0)",
            "fillColor": "rgba(0,0,0,0)",
        },
        "NOT_PREAPPROVED": {  # not yet pre-approved by Maha!
            "lineColor": "rgb(0,0,255)",
            "fillColor": "rgba(0,0,255,0.3)",
        }
    }
    for k in NOT_APPROVED:
        NOT_APPROVED[k]['group'] = k

    SUBSTRINGS = {
        'fov_preapproved_v1': ['fov_preapproved'],  # first preapproval cycle
        'fov_preapproved_v2': ['correction_fov'],  # second preapproval cycle
    }

# %%===========================================================================


def _get_participant_aliases(who_names):
    alias = {
        p: f'{grp[:-1]}.{i + 1}'
        for grp in ['SPs', 'JPs', 'NPs']
        for i, p in enumerate(who_names[grp])
    }
    extra_nps = ['Yahya_Mohammad', 'Abdelrahman_shahata', 'Mohamed_Amgad']
    for i, np in enumerate(extra_nps):
        alias[np] = f'NP.{len(who_names["NPs"]) + i + 1}'
    return alias


class Interrater(object):

    WHO_NAMES = {
        'SPs': ['Habiba_Elfandy', 'Hagar_Khallaf', 'Ehab_Hafiz'],
        'JPs': [
            'Kareem_Hosny', 'Liza_Hanna', 'Yan_Xiang', 'Mohamad_Gafeer',
            'Lamees_Atteya', 'Philip_Pappalardo',
        ],
        'NPs': [
            'Abdelmagid_Elmatboly', 'Abo-Alela_Farag', 'Ahmad_Rachid',
            'Ahmed_Afifi', 'Ahmed_Alhusseiny', 'Ahmed_Ayad',
            'Ahmed_Badr', 'Ahmed_Elkashash', 'Ahmed_Gadallah',
            'Ahmed_Gomaa', 'Ahmed_Raslan', 'Ali_Abdulkarim',
            'Amira_Etman', 'Anas_Alrefai', 'Anas_Saad',
            'Eman_Sakr', 'Esraa_Ghadban',
            'Inas_Ruhban',
            'Joumana_Ahmed',
            'Maha_Elsebaie', 'Mahmoud_Hashim', 'Menna_Nadeem',
            'Mohamed_Almoslemany', 'Mohamed_Hosny', 'Mohamed_Zalabia',
            'Nada_Elgazar',
            'Reham_Elfawal', 'Rokia_Sakr',
            'Yomna_Amer',
        ],
    }
    PARTICIPANT_ALIASES = _get_participant_aliases(WHO_NAMES)

    # get convenience variables -- everyone is anonymized
    tmp = {'SPs': [], 'JPs': [], 'NPs': []}
    for grp, als in tmp.items():
        for _, p in enumerate(WHO_NAMES[grp]):
            als.append(PARTICIPANT_ALIASES[p])
    SPs = tmp['SPs']
    JPs = tmp['JPs']
    NPs = tmp['NPs']
    Ps = SPs + JPs
    All = Ps + NPs
    who = {'SPs': SPs, 'JPs': JPs, 'Ps': Ps, 'NPs': NPs, 'All': All}

    # Who constitutes "truth" -- used to get sqlite table views once
    CONSENSUS_WHOS = ['Ps', 'NPs']
    MIN_DETECTIONS_PER_ANCHOR = 2
    CLASSES = DefaultAnnotationStyles.CLASSES

    ANNSETS = {
        'C': {
            'alias': 'CORE_SET',
            'long': 'Core Set'
        },
        'E': {
            'alias': 'EVAL_SET_3',
            'long': 'Main Evaluation Set'
        },
        'BT-control': {
            'alias': 'EVAL_SET_4',
            'long': 'Bootstrapped Color Thresholding Control Evaluation Set',
        },
        'B-control': {
            'alias': 'EVAL_SET_2',
            'long': 'Bootstrapped Control Evaluation Set',
        },
        'U-control': {
            'alias': 'EVAL_SET_1',
            'long': 'Unbiased Control Evaluation Set'
        },
    }
    REVERSE_ANNSETS = {
        v['alias']: {'alias': k, 'long': v['long']}
        for k, v in ANNSETS.items()
    }
    EVALSET_NAMES = ['U-control', 'BT-control', 'B-control', 'E']

    # Note: BT-control is useless because it was not annotated by enough Ps
    # .. only one JP annotated it, so all anchors are false there
    MAIN_EVALSET_NAMES = ['U-control', 'B-control', 'E']

    PARTICIPANT_STYLES = {
        'NPs': {'c': '#BB1AC7', 'c2': '#e591ff', 'marker': 'o'},
        'JPs': {'c': '#4f228a', 'c2': '#c7a4f5', 'marker': 's'},
        'SPs': {'c': '#17cbd1', 'c2': '#a6f2f5', 'marker': '^'},
        'Ps': {'c': '#17cbd1', 'c2': '#a6f2f5', 'marker': '^'},
    }

    EVALSET_STYLES = {
        'U-control': {'linestyle': 'dotted'},
        'B-control': {'linestyle': 'dashed'},
        'E': {'linestyle': 'solid'},
    }

    # Agreement ranges as described by Fleiss et al. Note that
    # Krippendorph's alpha is a generalization of the Kappa
    # statistics so these ranges are similar. Also note that these
    # ranges are not set in stone
    #  Fleiss, J. L. (1971) "Measuring nominal scale agreement among many
    #  raters." Psychological Bulletin, Vol. 76, No. 5 pp. 378–382
    KAPPA_RANGES = [
        {'descr': 'Slight', 'min': 0, 'max': 0.2},
        {'descr': 'Fair', 'min': 0.2, 'max': 0.4},
        {'descr': 'Moderate', 'min': 0.4, 'max': 0.6},
        {'descr': 'Substantial', 'min': 0.6, 'max': 0.8},
        {'descr': 'Near Perfect', 'min': 0.8, 'max': 1.},
    ]

    TRUTHMETHOD = 'EM'
    UNDETECTED_IS_A_CLASS = True

    MIN_KALPHA = 0.4
    KALPHA_PLUSMINUS = 0.05

    # IMPORTANT -- THESE THRESHOLDS ARE OBTAINED BY VISUAL INSPECTION OF THE
    # KRIPPENDORPH SUMMARY
    CMINIOU = 0.25  # clustering min iou
    CUTOFF_N_PATHOLOGISTS = -1  # min Ps per anchor to avoid exclusion

    # To facilitate sqlite querying
    @staticmethod
    def _get_sqlitestr_for_list(
            what: list, prefix: str = '', postfix: str = '') -> str:
        if isinstance(what, type({}.keys())):
            what = list(what)
        if what[0] == '*':
            return '*'
        return ','.join([f'{prefix}"{item}"{postfix}' for item in what])

    @staticmethod
    def _get_sqlite_usrstr_for_who(who: str) -> str:
        return Interrater._get_sqlitestr_for_list(what=Interrater.who[who])

    @staticmethod
    def _get_sqlite_truthstr(whoistruth=None, unbiased=None) -> str:
        ir = Interrater
        whoistruth = ir.CONSENSUS_WHOS if whoistruth is None else [whoistruth]
        unbiased = [True, False] if unbiased is None else [unbiased]
        return ','.join([
            f'"{ir._get_truthcol(whoistruth=who, unbiased=ub)}"'
            for who in whoistruth
            for ub in unbiased
        ])

    @staticmethod
    def _ubstr(unbiased: bool) -> str:
        return 'UNBIASED_' if unbiased else ''

    @staticmethod
    def _get_truthcol(whoistruth: str, unbiased: bool) -> str:
        ir = Interrater
        return f'{ir._ubstr(unbiased)}{ir.TRUTHMETHOD}_inferred_label_' \
               f'{whoistruth}'

    @staticmethod
    def _query_real_anchors_for_usr(
            dbcon, whoistruth: str, unbiased: bool, usr: str,
            evalset: str, colnames: Union[list, None] = None) -> DataFrame:
        ir = Interrater
        ubstr = ir._ubstr(unbiased)
        tablename = f'v3.1_final_anchors_' \
                    f'{evalset}_{ubstr}{whoistruth}_AreTruth'
        truthcol = ir._get_truthcol(whoistruth=whoistruth, unbiased=unbiased)
        if colnames is None:
            colnames = ['anchor_id', f'{usr}', f'{truthcol}']
        return read_sql_query(f"""
            SELECT {ir._get_sqlitestr_for_list(colnames)}
            FROM "{tablename}"
            WHERE "{usr}" != "DidNotAnnotateFOV"
        ;""", dbcon)

    @staticmethod
    def _query_fp_anchors_for_usr(
            dbcon, whoistruth: str, unbiased: bool, usr: str,
            evalset: str, colnames: Union[list, None] = None) -> DataFrame:
        ir = Interrater
        ubstr = ir._ubstr(unbiased)
        truthcol = ir._get_truthcol(whoistruth=whoistruth, unbiased=unbiased)
        common = f"""
            "min_iou" = {Interrater.CMINIOU}
            AND "{usr}" NOT IN ("DidNotAnnotateFOV", "undetected")
        """
        minn = ir.MIN_DETECTIONS_PER_ANCHOR
        if colnames is None:
            colnames = ['anchor_id', f'{usr}']
        return read_sql_query(f"""
            SELECT {ir._get_sqlitestr_for_list(colnames)}
            FROM "all_anchors_{evalset}"
            WHERE (
                  "{ubstr}n_matches_{whoistruth}" < {minn}
              AND {common}
            )
            OR (
                  "{ubstr}n_matches_{whoistruth}" >= {minn}
              AND "{truthcol}" = "undetected"
              AND {common}
            )
        ;""", dbcon)

    @staticmethod
    def _query_all_anchors_for_usr(
            dbcon, usr: str, evalset: str,
            get_clicks: bool = False) -> DataFrame:
        clicks = ', algorithmic_clicks_All' if get_clicks else ''
        return read_sql_query(f"""
            SELECT "anchor_id", "{usr}" {clicks}
            FROM "all_anchors_{evalset}"
            WHERE "{usr}" != "DidNotAnnotateFOV"
              AND "min_iou" = {Interrater.CMINIOU}
        ;""", dbcon)

    @staticmethod
    def _query_all_anchors_for_who(
            dbcon, who: str, evalset: str,
            get_truth: bool = False) -> DataFrame:
        ir = Interrater
        usrstr = ir._get_sqlite_usrstr_for_who(who)
        if get_truth:
            usrstr += ',' + ir._get_sqlite_truthstr()
        return read_sql_query(f"""
            SELECT "anchor_id", {usrstr}
            FROM "all_anchors_{evalset}"
            WHERE "min_iou" = {ir.CMINIOU}
        ;""", dbcon)

    @staticmethod
    def _get_true_and_inferred_labels_for_who(
            dbcon, whoistruth: str, unbiased: bool, who: str,
            evalset: str, colnames: Union[list, None] = None) -> DataFrame:
        ir = Interrater
        ubstr = ir._ubstr(unbiased)
        tablename = f'v3.1_final_anchors_' \
                    f'{evalset}_{ubstr}{whoistruth}_AreTruth'
        truthcol = ir._get_truthcol(
            whoistruth=whoistruth, unbiased=unbiased)
        if colnames is None:
            colnames = [
                'anchor_id', f'{truthcol}',
                f'EM_inferred_label_{who}',
                f'EM_inferred_label_confidence_{who}',
            ]
            colnames += [
                f'EM_prob_{cls}_{who}' for cls in ['undetected'] + ir.CLASSES]
        return read_sql_query(f"""
            SELECT {ir._get_sqlitestr_for_list(colnames)}
            FROM "{tablename}"
            WHERE "n_matches_{who}" >= 2 
              AND "EM_inferred_label_{who}" NOT NULL
        ;""", dbcon)

# %%===========================================================================
