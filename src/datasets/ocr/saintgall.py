from PIL import Image
import os
import json

from src.dataloaders.summed_dataloader import GenericDataset

DEFAULT_SAINT_GALL = "/data/users/amolina/OCR/SaintGall"

class SaintGallDataset(GenericDataset):
    def __init__(self) -> None:
        super().__init__()