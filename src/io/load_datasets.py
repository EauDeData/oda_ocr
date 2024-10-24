from src.datasets.ocr.esposalles import EsposalledDataset
from src.datasets.ocr.cocotext import COCOTextDataset
from src.datasets.ocr.funsd import FUNSDDataset
from src.datasets.ocr.washington import GWDataset
from src.datasets.ocr.hiertext import HierTextDataset
from src.datasets.ocr.historical_maps import HistoricalMapsdDataset
from src.datasets.ocr.iam import IAMDataset
from src.datasets.ocr.iit5k import IIIT5kDataset
from src.datasets.ocr.mlt19 import MLT19Dataset
from src.datasets.ocr.parzival import ParzivalDataset
from src.datasets.ocr.xfund import XFundDataset
from src.datasets.ocr.totaltext import TotalTextDataset
from src.datasets.ocr.textocr import TextOCRDataset
from src.datasets.ocr.svt import SVTDataset
from src.datasets.ocr.sroie import SROIEDataset
from src.datasets.ocr.saintgall import SaintGallDataset
from src.datasets.ocr.word_art import  WordArtDataset
from src.datasets.ocr.amr import AMRDataset
from src.datasets.ocr.copiale import CopialeDataset
from src.datasets.ocr.borg import BorgDataset
from src.datasets.ocr.vatican import VaticanDataset
def load_datasets(args, transforms=lambda x: x, split_langs=True):
    datasets = []

    common = {
        'image_height': args.image_height,
        'patch_width': args.patch_width,
        'transforms': transforms
    }

    bool_lut = {
        'true': True,
        'false': False
    }
    if args.use_word_art:
        datasets.append(
            {
                'train': WordArtDataset(base_location=args.word_art_path, split='train', **common),
                'val': WordArtDataset(base_location=args.word_art_path, split='validation', **common),
                'test': None
             }
        )
    if args.use_amr:
        datasets.append(
            {
                'train': AMRDataset(base_folder=args.amr_path, split='training', **common),
                'val': AMRDataset(base_folder=args.amr_path, split='validation', **common),
                'test': AMRDataset(base_folder=args.amr_path, split='testing', **common),
             }
        )

    if args.use_cocotext:

        if not split_langs:
            datasets.append({
                'train': COCOTextDataset(base_folder=args.cocotext_path, split='train', langs=args.cocotext_langs,
                                         legibility=args.cocotext_visibility, **common),
                'val': COCOTextDataset(base_folder=args.cocotext_path, split='val', langs=args.cocotext_langs,
                                       legibility=args.cocotext_visibility, **common),
                'test': None
            })
        else:
            for lang in args.cocotext_langs:
                datasets.append({
                    'train': COCOTextDataset(base_folder=args.cocotext_path, split='train', langs=[lang],
                                             legibility=args.cocotext_visibility, **common),
                    'val': COCOTextDataset(base_folder=args.cocotext_path, split='val', langs=[lang],
                                           legibility=args.cocotext_visibility, **common),
                    'test': None
                })

    if args.use_esposalles:
        datasets.append(
            {
                'train': EsposalledDataset(base_folder=args.esposalles_path, split='train',
                                           cross_val=args.esposalles_cross_validation_fold, mode=args.esposalles_level,
                                           **common),
                'test': EsposalledDataset(base_folder=args.esposalles_path, split='test',
                                          cross_val=args.esposalles_cross_validation_fold, mode=args.esposalles_level,
                                          **common),
                'val': None
            }
        )
    if args.use_funsd:
        datasets.append(
            {
                'train': FUNSDDataset(base_folder=args.funsd_path, split='train', **common),
                'test': FUNSDDataset(base_folder=args.funsd_path, split='test', **common),
                'val': None
            }
        )
    if args.use_hiertext:
        datasets.append(
            {
                'train': HierTextDataset(base_folder=args.hiertext_path, split='train',
                                         handwritten=[bool_lut[x] for x in args.hiertext_handwritten],
                                         legibility=[True] if not args.hiertext_include_non_visible else [True, False],
                                         mode=args.hiertext_level, **common),
                'val': HierTextDataset(base_folder=args.hiertext_path, split='val',
                                       handwritten=[bool_lut[x] for x in args.hiertext_handwritten],
                                       legibility=[True] if not args.hiertext_include_non_visible else [True, False],
                                       mode=args.hiertext_level, **common),
                'test': None
            }
        )

    if args.use_hist_maps:
        datasets.append(
            {
                'train': HistoricalMapsdDataset(base_folder=args.hist_maps_path, split='train',
                                                cross_val=args.hist_maps_cross_validation_fold, **common),
                'test': HistoricalMapsdDataset(base_folder=args.hist_maps_path, split='test',
                                               cross_val=args.hist_maps_cross_validation_fold, **common),
                'val': None
            }
        )

    if args.use_iam:
        datasets.append(
            {
                'train': IAMDataset(base_folder=args.iam_path, split='train', mode=args.iam_level, **common),
                'test': IAMDataset(base_folder=args.iam_path, split='test', mode=args.iam_level, **common),
                'val': IAMDataset(base_folder=args.iam_path, split='val', mode=args.iam_level, **common)
            }
        )

    if args.use_iiit:
        datasets.append(
            {
                'train': IIIT5kDataset(base_folder=args.iiit_path, split='train', **common),
                'test': IIIT5kDataset(base_folder=args.iiit_path, split='test', **common),
                'val': None
            }
        )

    if args.use_mlt:

        if not split_langs:
            datasets.append(
                {
                    'train': MLT19Dataset(base_folder=args.mlt_path, split='train', language=args.mlt19_langs,
                                          cross_val=args.mlt19_cross_validation_fold, **common),
                    'val': MLT19Dataset(base_folder=args.mlt_path, split='val', language=args.mlt19_langs,
                                        cross_val=args.mlt19_cross_validation_fold, **common),
                    'test': None
                }
            )
        else:
            for lang in args.mlt19_langs:
                datasets.append(
                    {
                        'train': MLT19Dataset(base_folder=args.mlt_path, split='train', language=[lang],
                                              cross_val=args.mlt19_cross_validation_fold, **common),
                        'val': MLT19Dataset(base_folder=args.mlt_path, split='val', language=[lang],
                                            cross_val=args.mlt19_cross_validation_fold, **common),
                        'test': None
                    }
                )
    if args.use_parzival:
        datasets.append(
            {
                'train': ParzivalDataset(base_folder=args.parzival_path, split='train', mode=args.parzival_level,
                                         **common),
                'val': ParzivalDataset(base_folder=args.parzival_path, split='valid', mode=args.parzival_level,
                                       **common),
                'test': ParzivalDataset(base_folder=args.parzival_path, split='test', mode=args.parzival_level,
                                        **common)

            }
        )

    if args.use_saint_gall:
        datasets.append(
            {
                'train': SaintGallDataset(base_folder=args.saint_gall_path, split='train', **common),
                'test': SaintGallDataset(base_folder=args.saint_gall_path, split='test', **common),
                'val': SaintGallDataset(base_folder=args.saint_gall_path, split='valid', **common)
            }
        )

    if args.use_sroie:
        datasets.append(
            {
                'train': SROIEDataset(base_folder=args.sroie_path, split='train', **common),
                'test': SROIEDataset(base_folder=args.sroie_path, split='test', **common),
                'val': None
            }
        )

    if args.use_svt:
        datasets.append(
            {
                'train': SVTDataset(base_folder=args.svt_path, split='train', **common),
                'test': SVTDataset(base_folder=args.svt_path, split='test', **common),
                'val': None

            }
        )

    if args.use_textocr:
        datasets.append(
            {
                'train': TextOCRDataset(base_folder=args.textocr_path, split='train', **common),
                'val': TextOCRDataset(base_folder=args.textocr_path, split='val', **common),
                'test': None
            }
        )

    if args.use_totaltext:
        datasets.append(
            {
                'train': TotalTextDataset(base_folder=args.totaltext_path, split='Train', **common),
                'test': TotalTextDataset(base_folder=args.totaltext_path, split='Test', **common),
                'val': None
            }
        )

    if args.use_washington:
        datasets.append(
            {
                'train': GWDataset(base_folder=args.washington_path, split='train',
                                   cross_val=args.washington_cross_validation_fold, mode=args.washington_level,
                                   **common),
                'test': GWDataset(base_folder=args.washington_path, split='test',
                                  cross_val=args.washington_cross_validation_fold, mode=args.washington_level,
                                  **common),
                'val': GWDataset(base_folder=args.washington_path, split='valid',
                                 cross_val=args.washington_cross_validation_fold, mode=args.washington_level, **common)
            }
        )

    if args.use_xfund:

        if not split_langs:
            datasets.append(
                {
                    'train': XFundDataset(base_folder=args.xfund_path, split='train', lang=args.xfund_langs, **common),
                    'test': None,
                    'val': XFundDataset(base_folder=args.xfund_path, split='val', lang=args.xfund_langs, **common),
                }
            )
        else:
            for lang in args.xfund_langs:
                datasets.append(
                    {
                        'train': XFundDataset(base_folder=args.xfund_path, split='train', lang=[lang],
                                              **common),
                        'test': None,
                        'val': XFundDataset(base_folder=args.xfund_path, split='val', lang=[lang], **common),
                    }
                )
    if args.use_copiale:

        datasets.append(
            {
                'train': CopialeDataset(base_folder=args.copiale_path, split='train', **common),
                'test': CopialeDataset(base_folder=args.copiale_path, split='test', **common),
                'val': CopialeDataset(base_folder=args.copiale_path, split='valid', **common)

            }
        )
    if args.use_borg:

        datasets.append(
            {
                'train': BorgDataset(base_folder=args.borg_path, split='train', **common),
                'test': BorgDataset(base_folder=args.borg_path, split='test', **common),
                'val': BorgDataset(base_folder=args.borg_path, split='valid', **common)

            }
        )
    if args.use_vatican:

        datasets.append(
            {
                'train': VaticanDataset(base_folder=args.vatican_path, split='train', **common),
                'test': VaticanDataset(base_folder=args.vatican_path, split='test', **common),
                'val': VaticanDataset(base_folder=args.vatican_path, split='valid', **common)

            }
        )
    return datasets

def log_usage(args, split_langs=False):
    datasets = []

    if args.use_cocotext:

        if not split_langs:
            datasets.append('cocotext')
        else:
            for lang in args.cocotext_langs:
                datasets.append(f"cocotext_{lang}")

    if args.use_esposalles:
        datasets.append('esposalles')

    if args.use_funsd:
        datasets.append('funsd')

    if args.use_hiertext:
        datasets.append('hiertext')

    if args.use_hist_maps:
        datasets.append('hist_maps')

    if args.use_iam:
        datasets.append('iam')

    if args.use_iiit:
        datasets.append('iiit')

    if args.use_mlt:

        if not split_langs:
            datasets.append('mlt')
        else:
            for lang in args.mlt19_langs:
                datasets.append(f"mlt_{lang}")
    if args.use_parzival:
        datasets.append('parzival')

    if args.use_saint_gall:
        datasets.append('saintgall')

    if args.use_sroie:
        datasets.append('sroie')

    if args.use_svt:
        datasets.append('svt')

    if args.use_textocr:
        datasets.append('textocr')

    if args.use_totaltext:
        datasets.append('totaltext')

    if args.use_washington:
        datasets.append('gw')

    if args.use_xfund:

        if not split_langs:
            datasets.append('xfund')
        else:
            for lang in args.xfund_langs:
                datasets.append(f"xfund_{lang}")

    return datasets
