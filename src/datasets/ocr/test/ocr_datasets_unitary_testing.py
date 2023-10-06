from src.datasets.ocr.esposalles import EsposalledDataset, DEFAULT_ESPOSALLES
from src.datasets.ocr.cocotext import COCOTextDataset, DEFAULT_COCOTEXT
from src.datasets.ocr.funsd import FUNSDDataset, DEFAULT_FUNSD


def try_esposalles(base_folder = DEFAULT_ESPOSALLES, split = 'train', cross_val = 'cv1', mode = 'words', image_height = 128, patch_width = 16, transforms = lambda x: x):

    dataset = EsposalledDataset(base_folder, split, cross_val, mode, image_height, patch_width, transforms)
    print(dataset[0])
    print('total:', len(dataset))
    
def try_cocotext(base_folder = DEFAULT_COCOTEXT, annots_name='cocotext.v2.json', langs = ['english', 'non-english'], legibility = ['legible', 'illgible'], split = 'train', image_height = 128, patch_width = 16, transforms = lambda x: x):
    dataset = COCOTextDataset(base_folder, annots_name, split, langs, legibility, image_height, patch_width, transforms )

    print('total:', len(dataset))
    print(dataset[0])
    
def try_funsd(base_folder = DEFAULT_FUNSD, split: ['train', 'test'] = 'train', patch_width = 16, image_height = 128, transformations = lambda x: x):
    
    dataset = FUNSDDataset(base_folder, split, patch_width, image_height, transformations)
    print('total:', len(dataset))
    print(dataset[0])