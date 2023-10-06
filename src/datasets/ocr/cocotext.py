from PIL import Image
import os
import json

from src.dataloaders.summed_dataloader import GenericDataset

DEFAULT_COCOTEXT = "/data/users/amolina/OCR/COCOText"

class COCOTextDataset(GenericDataset):
    def __init__(self, base_folder = DEFAULT_COCOTEXT, annots_name='cocotext.v2.json', split = 'train', image_height = 128, patch_width = 16, transforms = lambda x: x) -> None:
        super().__init__()
        
        json_annots = json.load(
            open(os.path.join(base_folder, annots_name))
        )
        print(json_annots)