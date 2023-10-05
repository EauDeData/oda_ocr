from PIL import Image
import os

from src.dataloaders.summed_dataloader import GenericDataset

DEFAULT_ESPOSALLES = "/data/users/amolina/OCR/ESPOSALLES"

class EsposalledDataset(GenericDataset):
    name = 'esposalles_dataset'

    def __init__(self, base_folder = DEFAULT_ESPOSALLES, split = 'train', cross_val = 'cv1', mode = 'words', image_height = 128, patch_width = 16) -> None:
        super().__init__()
        #### ESPOSALLES DATASET ####
        # base_folder: folder with train - test splits.
        # split: train or test
        # cross-val: cross validation split
        # mode: words or lines.
        # image_height: height of the image sequence. Resize to this height.
        # patch_width: width of the sliding window. Resize sequence to closest superior multiple.
        ############################

        self.image_height = image_height
        self.patch_width = patch_width

        self.base_folder = os.path.join(base_folder, split)
        valid_records = None # Valid records from cross validation fold

        records = [os.path.join(self.base_folder, page_id, mode) for page_id in os.listdir(self.base_folder) if page_id in valid_records]