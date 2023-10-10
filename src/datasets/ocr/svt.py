from PIL import Image
import os
import json

from src.dataloaders.summed_dataloader import GenericDataset

DEFAULT_SVT = "/data/users/amolina/OCR/SVT"

class SVTDataset(GenericDataset):
    def __init__(self) -> None:
        super().__init__()
