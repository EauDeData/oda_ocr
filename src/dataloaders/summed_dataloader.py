import numpy as np
from PIL import Image
import torch


class GenericDataset:

    def __init__(self):
        self.patch_width = None
        self.image_height = None

    def add(self, dataset):
        return SummedDataset(self, dataset)

    def resize_image(self, image):
        original_width, original_height = image.size

        original_height = max(original_height, 1)
        original_width = max(original_width, 1)

        scale = self.image_height / original_height

        resized_width = int(round(scale * original_width, 0))
        new_width = resized_width + (self.patch_width - (resized_width % self.patch_width))  # Adjusted this line

        return image.resize((new_width, self.image_height))

    def __add__(self, dataset):
        return self.add(dataset)


class SummedDataset(GenericDataset):
    def __init__(self, dataset_left, dataset_right) -> None:
        self.left = dataset_left
        self.right = dataset_right

    def __len__(self):
        return len(self.left) + len(self.right)

    def __getitem__(self, idx):
        if idx > (len(self.left) - 1):
            idx_corrected = idx % len(self.left)
            return self.right[idx_corrected]

        return self.left[idx]


class DummyDataset(GenericDataset):

    def __init__(self, number=30, name='dummy_v1', split='test') -> None:
        self.samples = list(range(number))
        self.name = name
        self.split = split

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return {
            'dataset_name': self.name,
            'split': self.split,
            'image': self.samples[idx]
        }

class CollateFNs:

    def __init__(self, patch_width, image_height, character_tokenizer, seq2seq_same_size=False, max_size=224,
                 transforms=lambda x: x) -> None:
        self.patch_width = patch_width
        self.image_height = image_height
        self.character_tokenizer = character_tokenizer
        self.visual_padding_token = torch.zeros((3, self.image_height, self.patch_width))
        self.total_visual_padding = torch.zeros((3, max_size, max_size))

        self.same_size = seq2seq_same_size
        self.max_size = max_size
        self.transforms = transforms

    def collate(self, batch):
        max_tokens = max([len(x['tokens']) for x in batch])
        max_patches = max([x['input_tensor'].shape[2] // self.patch_width for x in batch])
        max_tokens_both = max(max_patches, max_tokens)

        visual_tokens = []
        text_tokens = []
        raw_texts = []
        sources = []
        resized_images = []
        original_images = []
        images_tensor_full = []
        patched_images = []
        masks = []
        images_full_resized = []

        for item in batch:
            original_image, image, text_token, raw_text, split, dataset, resized_image = item['original_image'], item[
                'input_tensor'], item['tokens'], item['annotation'], item['split'], item['dataset'], item[
                'resized_image']
            original_images.append(original_image)

            resized_image = resize_to_max_size(resized_image, self.max_size)
            image_to_be_patched = self.transforms(resized_image)
            images_full_resized.append(self.transforms(resized_image.resize((self.max_size, self.max_size))))

            _, w, h = image_to_be_patched.shape

            final_image = self.total_visual_padding.clone()
            # mask = self.total_visual_padding.clone()

            final_image[:, :w, :h] = image_to_be_patched
            # mask[:, w:, h:] = 1
            # mask = 1 - mask

            # masks.append(mask)
            patched_images.append(final_image)

            resized_images.append(resized_image)
            sources.append(f"{split}_ {dataset}")
            raw_texts.append(raw_text.lower())

            patches = list(image.chunk(image.shape[2] // self.patch_width, dim=-1))
            patches = patches + [self.visual_padding_token] * (max_tokens_both - len(patches))

            text_tokenized = torch.from_numpy(
                self.character_tokenizer(
                    text_token + [self.character_tokenizer.padding_token] * (max_tokens_both - len(text_token))
                )
            )

            text_tokens.append(text_tokenized)

            visual_tokens.append(
                torch.stack(
                    patches
                )
            )
            images_tensor_full.append(
                torch.cat(patches, dim=2)
            )

        return {
            'input_visual_seq': torch.stack(visual_tokens),
            'images_tensor': torch.stack(images_tensor_full),
            'labels': torch.stack(text_tokens),
            'raw_text_gt': raw_texts,
            'sources': sources,
            'original_images': resized_images,
            'input_lengths': [x.size[0] // self.patch_width for x in resized_images],
            'output_lengths': [len([char for char in x]) for x in raw_texts],
            'pre_resize_images': original_images,
            'totally_padded_image': torch.stack(patched_images),
            'input_lengths_clip': [224 // self.patch_width for _ in resized_images],
            'masks': masks,
            'square_full_images': torch.stack(images_full_resized)
        }

def resize_to_max_size(image, max_size):
    width, height = image.size
    return image.resize((min(max_size, width), min(max_size, height)))


if __name__ == '__main__':
    dataset_0_3 = DummyDataset(3, name='dataset_1', split='test')
    dataset_3_5 = DummyDataset(2, name='dataset_2', split='test')
    dataset_5_10 = DummyDataset(5, name='dataset_3', split='val')

    dataset_0_5 = dataset_0_3 + dataset_3_5
    dataset_0_10 = dataset_0_5 + dataset_5_10

    print(dataset_0_10[0])  # Prints dataset_1: 0
    print(dataset_0_10[3])  # Prints dataset_2: 0
    print(dataset_0_10[5])  # Prints dataset_3: 0

    print(dataset_0_10[2])  # Prints dataset_1: 2
    print(dataset_0_10[4])  # Prints dataset_2: 1
    print(dataset_0_10[9])  # Prints dataset_3: 4
