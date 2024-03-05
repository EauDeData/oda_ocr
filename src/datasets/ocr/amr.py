from PIL import Image
import os
import json

from src.dataloaders.summed_dataloader import GenericDataset

DEFAULT_AMR = "/data2/users/amolina/OCR/AMR"

class AMRDataset(GenericDataset):
    name = 'amr_dataset'

    def __init__(self, base_folder=DEFAULT_AMR, split: ['training', 'testing', 'validation'] = 'training',
                 patch_width=16, image_height=128,
                 transforms=lambda x: x) -> None:
        super().__init__()
        self.data = [
            (os.path.join(base_folder, split, file),
             os.path.join(base_folder, split, file.replace('.jpg', '.txt'))
             )
            for file in
            os.listdir(
                os.path.join(base_folder, split)
            ) if file.endswith('.jpg') ]

        self.transforms = transforms
        self.split = split
        self.image_height = image_height
        self.patch_width = patch_width

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        file, transcription = self.data[item]
        lines = [x.strip().split(': ')[-1] for x in open(transcription).readlines()]
        gt, position = lines[1], lines[2]
        x,y,w,h = (int(x) for x in position.split())
        image = Image.open(

            file

        ).crop((x, y, x+w, y+h)).convert('RGB')
        image_resized = self.resize_image(image)

        input_tensor = self.transforms(image_resized)

        return {
            "original_image": image,
            "resized_image": image_resized,
            "input_tensor": input_tensor,
            "annotation": gt,
            'dataset': self.name,
            'split': self.split,
            'tokens': [char for char in gt]

        }
