from PIL import Image
import os
import json

from src.dataloaders.summed_dataloader import GenericDataset

DEFAULT_SROIE = "/data/users/amolina/OCR/SROIE"

class SROIEDataset(GenericDataset):
    def __init__(self) -> None:
        super().__init__()