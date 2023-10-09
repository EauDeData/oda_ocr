import os

from src.datasets.ocr.esposalles import EsposalledDataset, DEFAULT_ESPOSALLES
from src.datasets.ocr.cocotext import COCOTextDataset, DEFAULT_COCOTEXT
from src.datasets.ocr.funsd import FUNSDDataset, DEFAULT_FUNSD
from src.datasets.ocr.washington import GWDataset, DEFAULT_WASHINGTON
from src.datasets.ocr.hiertext import HierTextDataset, DEFAULT_HIERTEXT
from src.datasets.ocr.historical_maps import HistoricalMapsdDataset, DEFAULT_HIST_MAPS
from src.datasets.ocr.iam import IAMDataset, DEFAULT_IAM


IDX = 42
OUTPUT_TMP_FOLDER = './tmp_/'
os.makedirs(OUTPUT_TMP_FOLDER, exist_ok=True)

def log_dataset(dataset, idx = IDX, image_to_observe = 'original_image'):
    print('total:', len(dataset))
    out = dataset[IDX]
    print(out)
    out[image_to_observe].save(os.path.join(OUTPUT_TMP_FOLDER, f"{out['dataset']}_{out['split']}.png"))

def try_esposalles(base_folder = DEFAULT_ESPOSALLES, split = 'train', cross_val = 'cv1', mode = 'words', image_height = 128, patch_width = 16, transforms = lambda x: x):

    dataset = EsposalledDataset(base_folder, split, cross_val, mode, image_height, patch_width, transforms)
    log_dataset(dataset)
    
def try_cocotext(base_folder = DEFAULT_COCOTEXT, annots_name='cocotext.v2.json', langs = ['english', 'non-english'], legibility = ['legible', 'illgible'], split = 'train', image_height = 128, patch_width = 16, transforms = lambda x: x):
    dataset = COCOTextDataset(base_folder, annots_name, split, langs, legibility, image_height, patch_width, transforms )
    log_dataset(dataset)


    
def try_funsd(base_folder = DEFAULT_FUNSD, split: ['train', 'test'] = 'train', patch_width = 16, image_height = 128, transformations = lambda x: x):
    
    dataset = FUNSDDataset(base_folder, split, patch_width, image_height, transformations)
    log_dataset(dataset)

    
def try_washinton(base_folder = DEFAULT_WASHINGTON, split: ['train', 'test', 'val'] = 'train', cross_val = 'cv1', mode = 'word', image_height = 128, patch_width = 16, transforms = lambda x: x):
    
    dataset = GWDataset(base_folder, split, cross_val, mode, image_height, patch_width, transforms)
    log_dataset(dataset)

    
def try_hiertext(base_folder = DEFAULT_HIERTEXT, split: ['train', 'val'] = 'train',
                 handwritten = [True, False], legibility = [True, False], mode = 'words',
                 image_height = 128, patch_width = 16, transforms = lambda x: x):
    
    dataset = HierTextDataset(base_folder, split,
                 handwritten, legibility , mode ,
                 image_height ,patch_width , transforms)
    log_dataset(dataset)


def try_maps(base_folder = DEFAULT_HIST_MAPS, split: ["train", "test"] = 'train', cross_val = 'cv1', image_height = 128, patch_width = 16, transforms = lambda x: x):
    dataset =  HistoricalMapsdDataset(base_folder, split, cross_val, image_height, patch_width, transforms)
    log_dataset(dataset)


def try_iam(base_folder = DEFAULT_IAM, split: ["train", "test", "val"] = 'train', partition: ["aachen", "original"] = 'aachen', mode: ["words", "lines"] = "words", image_height = 128, patch_width = 16, transforms = lambda x: x):
    dataset = IAMDataset(base_folder, split, partition, mode, image_height, patch_width, transforms)
    log_dataset(dataset)

