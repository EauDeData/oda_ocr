from PIL import Image
import os
import json

from src.datasets.ocr.vatican import VaticanDataset
DEFAULT_COPIALE = "/data2/users/amolina/OCR/Borg/"

class CopialeDataset(VaticanDataset):
    name = 'copiale_dataset'
    def __init__(self, base_folder = DEFAULT_COPIALE,
                 split: ['train', 'test', 'valid'] = 'train',
                 image_height = 128, patch_width = 16, transforms = lambda x: x) -> None:
        super(CopialeDataset, self).__init__(base_folder, split, image_height, patch_width, transforms)


