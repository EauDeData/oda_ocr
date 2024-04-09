from src.datasets.ocr.vatican import VaticanDataset
DEFAULT_BORG = "/data2/users/amolina/OCR/Borg/"

class BorgDataset(VaticanDataset):
    name = 'borg_dataset'
    def __init__(self, base_folder = DEFAULT_BORG,
                 split: ['train', 'test', 'valid'] = 'train',
                 image_height = 128, patch_width = 16, transforms = lambda x: x) -> None:
        super(BorgDataset, self).__init__(base_folder, split, image_height, patch_width, transforms)
