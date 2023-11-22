from src.datasets.ocr.test.ocr_datasets_unitary_testing import (log_dataset, try_esposalles, try_cocotext, try_funsd,
                                                                try_washinton, try_hiertext, try_maps, try_iam, \
                                                                try_iii, try_mlt19, try_parzival, try_xfund,
                                                                try_totaltext, try_textocr, try_svt, try_sroie,
                                                                try_saint_gall)

from src.io.load_datasets import load_datasets
from src.io.args import parse_arguments
from torch.utils.data import DataLoader
from src.dataloaders.summed_dataloader import  CollateFNs
from src.tokenizers.char_tokenizer import CharTokenizer
import torchvision

args = parse_arguments()


normalize = {
    'normalize': lambda x: (x - x.min()) / max((x.max() - x.min()), 0.01),
    'standarize': lambda x: x / max(x.max(), 1)
}

transforms = torchvision.transforms.Compose((
    torchvision.transforms.PILToTensor(),
    normalize['normalize' if not args.standarize else 'standarize'])
)

datasets = load_datasets(args, transforms)

general_set = None
for dataset in datasets:
    for split in ['train', 'test', 'val']:
        if dataset[split] is None: continue
        if general_set is None: general_set = dataset[split]
        else: general_set = general_set + dataset[split]

tokenizer = CharTokenizer(tokenizer_name='oda_giga_tokenizer')
q = len(general_set)

collator = CollateFNs(args.patch_width, args.image_height, tokenizer, transforms=transforms)

for i, batch in enumerate(DataLoader(general_set, num_workers=32, collate_fn=collator.collate_while_debugging, batch_size=10)):

    print(f"{i} / {q}\t", end = '\r')

