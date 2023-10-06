from src.datasets.ocr.esposalles import EsposalledDataset, DEFAULT_ESPOSALLES
from src.datasets.ocr.cocotext import COCOTextDataset, DEFAULT_COCOTEXT

def try_esposalles(base_folder = DEFAULT_ESPOSALLES, split = 'train', cross_val = 'cv1', mode = 'words', image_height = 128, patch_width = 16, transforms = lambda x: x):
    #### ESPOSALLES DATASET TEST ####
    ###### dataset init #######
    dataset = EsposalledDataset(base_folder, split, cross_val, mode, image_height, patch_width, transforms)
    print(dataset[0])
    
def try_cocotext(base_folder = DEFAULT_COCOTEXT, annots_name='cocotext.v2.json', split = 'train', image_height = 128, patch_width = 16, transforms = lambda x: x):
    pass