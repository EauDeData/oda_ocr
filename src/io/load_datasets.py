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


## AQUI AGAFAR ELS ARGUMENTS I ANAR FENT DICCIONARIS 

{'train': None,
 'val': None,
 'test': None}

# Amb None si no está disponible i el dataset si ho està


def load_datasets(args, transforms = lambda x: x):
    datasets = []

    common = {
        'image_height': args.image_height,
        'patch_with': args.patch_width,
        'transforms': transforms
    }

    bool_lut = {
        'true': True,
        'false': False
    }

    if args.use_cocotext:

        datasets.append({
            'train': COCOTextDataset(base_folder=args.cocotext_path, split = 'train', langs=args.cocotext_langs, legibility=args.cocotext_visibility, **common),
            'train': COCOTextDataset(base_folder=args.cocotext_path, split = 'val', langs=args.cocotext_langs, legibility=args.cocotext_visibility, **common),
            'test': None
        })

    if args.use_esposalles:

        datasets.append(
            {
                'train': EsposalledDataset(base_folder=args.esposalles_path, split = 'train', cross_val=args.esposalles_cross_validation_fold, mode = args.esposalles_level, **common),
                'test': EsposalledDataset(base_folder=args.esposalles_path, split = 'test', cross_val=args.esposalles_cross_validation_fold, mode = args.esposalles_level, **common),
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
                'train': HierTextDataset(base_folder=args.hiertext_path, split = 'train', handwritten=[bool_lut[x] for x in args.hiertext_handwritten], legibility=[True] if not args.hiertext_include_non_visible else [True, False], mode = args.hiertext_level, **common),
                'val': HierTextDataset(base_folder=args.hiertext_path, split = 'val', handwritten=[bool_lut[x] for x in args.hiertext_handwritten], legibility=[True] if not args.hiertext_include_non_visible else [True, False], mode = args.hiertext_level, **common),
                'test': None
            }
        )

    if args.use_hist_maps:

        datasets.append(
            {
                'train': HistoricalMapsdDataset(base_folder=args.hist_maps_path, split = 'train', cross_val=args.hist_maps_cross_validation_fold, **common),
                'test': HistoricalMapsdDataset(base_folder=args.hist_maps_path, split = 'train', cross_val=args.hist_maps_cross_validation_fold, **common),
                'val': None
            }
        )
    # TODO: Amb calma i en ordre, ves fent d'adalt abaix...