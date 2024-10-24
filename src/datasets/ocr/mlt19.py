from PIL import Image
import os
import json 

from src.dataloaders.summed_dataloader import GenericDataset

DEFAULT_MLT = "/data2/users/amolina/OCR/MLT19/"

class MLT19Dataset(GenericDataset):

    def __init__(self, base_folder = DEFAULT_MLT, split: ["train", "val"] = 'train',\
                 language = ['Latin', 'Arabic',  "Chinese", "Japanese", "Korean", "Bangla", "Hindi", "Symbols", "Mixed", "None"],\
                cross_val = 'cv1', image_height = 128, patch_width = 16, transforms = lambda x: x) -> None:

        self.name = f"mlt19_dataset_{'_'.join([x[:3] for x in language])}"

        self.image_folder = os.path.join(
            base_folder, 'images'
        )

        self.annotations_folder = os.path.join(
            base_folder, 'gt'
        )

        self.split = f"{split}_{cross_val}"
        self.image_height = image_height
        self.patch_width = patch_width
        self.transforms = transforms

        valid_anns = [x.strip() for x in open(
            os.path.join(
                base_folder, 'cv_splits', cross_val, split + '.txt'
            ), 'r'
        ).readlines()]

        self.data = []
        for gt_file in os.listdir(self.annotations_folder):
            if gt_file in valid_anns:
                image_path = os.path.join(
                    self.image_folder, gt_file.replace('.txt', '.jpg')
                )
                for line in open(os.path.join(self.annotations_folder, gt_file)).readlines():
                    
                    parts = line.strip().split(',')
                    unfolded_parts = parts[:9] + [','.join(parts[9:])]
                    xx, xy, yy, yx, x2x, x2y, y2y, y2x, lang, transcription = unfolded_parts

                    if lang in language:

                        points = [[float(y) for y in x] for x in [[xx, xy],
                                  [yy, yx],
                                  [x2x, x2y],
                                  [y2y, y2x]]]
                        
                        if lang != 'Latin' and (not ('#' in transcription)):
                            self.data.append({

                                'image_path': image_path,
                                'transcription': transcription,
                                'bbx': (min(points, key = lambda x: x[0])[0], min(points, key = lambda x: x[1])[1], max(points, key = lambda x: x[0])[0], max(points, key = lambda x: x[1])[1])

                                }
                            )
                        elif lang == 'Latin':
                            # Temporal, for some reproducibility in a single experiment
                            self.data.append({

                                'image_path': image_path,
                                'transcription': transcription,
                                'bbx': (min(points, key=lambda x: x[0])[0], min(points, key=lambda x: x[1])[1],
                                        max(points, key=lambda x: x[0])[0], max(points, key=lambda x: x[1])[1])

                            }
                            )
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        
        metadata = self.data[idx]
        x,y,w,h = metadata['bbx']
        image = Image.open(
                           
                           metadata['image_path']
                           
                           )
        
        w,h = min(w, image.size[0]), min(h, image.size[1])
        image = image.crop((x, y, w , h )).convert('RGB')
        
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