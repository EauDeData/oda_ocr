from PIL import Image
import os
import json

from src.dataloaders.summed_dataloader import GenericDataset

DEFAULT_XFUND = "/data2/users/amolina/OCR/xFUND"

class XFundDataset(GenericDataset):

    def __init__(self, base_folder = DEFAULT_XFUND, split: ['train', 'val'] = 'train', lang = ['DE', 'ES', 'FR', 'IT', 'JA', 'PT', 'ZH'], image_height = 128, patch_width = 16, transforms = lambda x: x) -> None:

        self.name = f"xfund_dataset_{'_'.join([x for x in lang])}"

        self.split = split
        self.image_height = image_height
        self.patch_width = patch_width

        self.transforms = transforms
        self.data = []

        for lan_id in lang:

            base_lang_folder = os.path.join(
                base_folder, lan_id
            )

            annotations = json.load(
                open(
                        os.path.join(
                    base_lang_folder, f"{lan_id.lower()}.{split}.json"
                                    ), 'r'
                    )
                )

            for document in annotations['documents']:

                page_path = os.path.join(
                    base_lang_folder, f"{lan_id.lower()}.{split}", document['id']+'.jpg'
                )

                for item in document['document']:

                    bbx = item['box']
                    transcription = item['text']

                    self.data.append(
                        {
                            'transcription': transcription,
                            'bbx': bbx,
                            'image_path': page_path
                        }
                    )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        
        metadata = self.data[idx]
        x,y,w,h = metadata['bbx']

        image = Image.open(
                           
                           metadata['image_path']
                           
                           ).crop((x, y, w, h)).convert('RGB')
        
        image_resized = self.resize_image(image)

        input_tensor = self.transforms(image_resized)
        
        return {
            "original_image": image,
            "resized_image": image_resized,
            "input_tensor": input_tensor,
            "annotation": metadata['transcription'],
            'dataset': self.name,
            'split': self.split,
            'tokens': [char for char in metadata['transcription']]

        }



