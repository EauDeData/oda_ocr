import os
import torch, torchvision
import numpy as np
import wandb
import torch.optim as optim
from tqdm import tqdm
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from main import prepare_tokenizer_and_collator, merge_datasets, prepare_train_loaders
from src.io.args import parse_arguments
from src.io.load_datasets import load_datasets
from src.dataloaders.summed_dataloader import CollateFNs
from src.tokenizers.char_tokenizer import CharTokenizer

def get_sizes(dataset):
    heights = []
    widths = []
    collections = []
    
    for batch in tqdm(dataset, desc = 'obtaining sizes for posterior plot...'):
        for image, label in zip(batch['pre_resize_images'], batch['sources']):
            
            width, height = image.size
            
            heights.append(height)
            widths.append(width)
            
            collections.append(label)

    return pd.DataFrame({
        "height": heights,
        "width": widths,
        "collections": collections 
    })

def main_plot(args):
        
    transforms = torchvision.transforms.PILToTensor()

    
    print('Loading all datasets...')
    datasets = load_datasets(args, transforms)
    print(f"Loaded {len(datasets)} datasets")
    whole_train = merge_datasets(datasets, split = 'train')
    
    tokenizer, collator = prepare_tokenizer_and_collator(whole_train, args)
    train_dataloader = prepare_train_loaders(whole_train, collator, args.num_workers_train, args.batch_size)

    dataframe = get_sizes(train_dataloader)
    sns.scatterplot(data = dataframe, x = 'width', y = 'height', hue = 'collections')
    plt.grid(True)
    plt.savefig('tmp_/sizes.png', transparent = True)
     

if __name__ == '__main__':
    
    args = parse_arguments()
    main_plot(args)