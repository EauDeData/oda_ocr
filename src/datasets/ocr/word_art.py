from PIL import Image
import os
import json
from src.dataloaders.summed_dataloader import GenericDataset

DEFAULT_WA = "/data/users/amolina/OCR/WordArt"

class WordArtDataset(GenericDataset):
    name = 'word_art_dataset'
    def __init__(self, split = 'train', base_location=DEFAULT_WA, transforms=lambda x: x, image_height=0, patch_width=0):
        self.paths = [x for x in [
            (os.path.join(base_location, line[0].strip().replace('\\', '/')), line[1]) for line in
            [L.strip().split(' ') for L in open(
                os.path.join(base_location, f"{split}.txt")
            ).readlines()]
        ] if os.path.exists(x[0])]

        self.transforms = transforms
        self.split = split
        self.image_height = image_height
        self.patch_width = patch_width


    def __len__(self):
        return len(self.paths)

    def __getitem__(self, item):

        file, transcription = self.paths[item]
        image = Image.open(

            os.path.join(file)

        ).convert('RGB')

        image_resized = self.resize_image(image)

        input_tensor = self.transforms(image_resized)

        return {
            "original_image": image,
            "resized_image": image_resized,
            "input_tensor": input_tensor,
            "annotation": transcription,
            'dataset': self.name,
            'split': self.split,
            'tokens': [char for char in transcription]

        }