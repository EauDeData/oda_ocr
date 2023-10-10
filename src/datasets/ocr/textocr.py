from PIL import Image
import os
import json

from src.dataloaders.summed_dataloader import GenericDataset

DEFAULT_TEXTOCR = "/data/users/amolina/OCR/TextOCR"

class TextOCRDataset(GenericDataset):
    def __init__(self) -> None:
        super().__init__()