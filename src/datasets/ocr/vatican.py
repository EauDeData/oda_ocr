from PIL import Image
import os
import json

from src.dataloaders.summed_dataloader import GenericDataset

DEFAULT_VATICAN = "/data2/users/amolina/OCR/Vatican/"

class VaticanDataset(GenericDataset):
    name = 'vatican_dataset'
    def __init__(self, base_folder = DEFAULT_VATICAN,
                 split: ['train', 'test', 'valid'] = 'train',
                 image_height = 128, patch_width = 16, transforms = lambda x: x) -> None:
        super().__init__()

        self.transforms = transforms
        self.split = split
        self.image_height = image_height
        self.patch_width = patch_width
        self.img_folder = os.path.join(base_folder, 'words')
        transcriptions_file = os.path.join(base_folder, f"gt_{split}.txt")
        self.mode = split
        self.transcriptions = list()
        for transcription in open(transcriptions_file).readlines():

            filename, tokens_line = transcription.strip().split('|')
            tokens = [x for x in tokens_line.split(' ') if len(x)]
            if len(tokens):
                self.transcriptions.append((os.path.join(self.img_folder, filename), tokens))

    def __len__(self):
        return len(self.transcriptions)
    def __getitem__(self, idx):

        filename, annotation = self.transcriptions[idx]

        image = Image.open(filename).convert('RGB')

        image_resized = self.resize_image(image)

        input_tensor = self.transforms(image_resized)


        return {
            "original_image": image,
            "resized_image": image_resized,
            "input_tensor": input_tensor,
            "annotation": ''.join(annotation),
            'dataset': self.name,
            'split': f"{self.mode}_{self.split}",
            'path': filename,
            'tokens': annotation
        }


