import os
import torch, torchvision
import numpy as np
import wandb

from src.io.args import parse_arguments, get_model_name
from src.io.load_datasets import load_datasets
from src.dataloaders.summed_dataloader import CollateFNs
from src.tokenizers.char_tokenizer import CharTokenizer
from src.vision.models import ViTEncoder
from src.linearize import LinearizedModel

def merge_datasets(datasets, split = 'train'):
    
    data = datasets[0][split]
    
    for idx in range(1, len(datasets)):
        
        data = data + datasets[idx][split]
        
    return data

def prepare_tokenizer_and_collator(merged_dataset, args):
  
  tokenizer = CharTokenizer(merged_dataset, args.tokenizer_location, args.tokenizer_name, args.save_tokenizer)
  collator = CollateFNs(args.patch_width, args.image_height, tokenizer)
  
  return tokenizer, collator

def prepare_train_loaders(dataset, collator, num_workers, batch_size):
    return torch.utils.data.DataLoader(dataset, batch_size = batch_size, collate_fn = collator.collate, num_workers = num_workers, shuffle = True)

def prepare_model(vocab_size, args):
    #### LOAD MODEL ###
    if args.load_checkpoint:
        raise NotImplementedError(f"Won't load {args.checkpoint_name}, model loading is not implemented yet.")
    
    else:
        model = ViTEncoder(args.image_height, args.patch_width, 3, args.token_size, [args.visual_tokenizer_width] * args.visual_tokenizer_depth, args.model_depth, args.model_width, vocab_size, args.dropout, args.device)
        model.to(args.device)

    ### LINEARIZE ###
    ### The loaded model is already linear?
    if args.linear_model:
        print('Linearizing ViT model...')
        model = LinearizedModel(model)
    
    return model

# We will heve to define training strategies for "simple", "continual" and "arithmetic".
## Arithmetic models are trained normally, but they are linearized with the modules

def main(args):
    
    model_name = get_model_name(args)
    print(model_name)
    wandb.config.update(args)
    wandb.run.name = model_name
    
    normalize = {
        'normalize': lambda x: (x - x.min()) / (x.max() - x.min()),
        'standarize': lambda x: x / x.max()
    }
    
    transforms = torchvision.transforms.Compose((
        torchvision.transforms.PILToTensor(),
        normalize['normalize' if not args.standarize else 'standarize']) 
    )
    
    print('Loading all datasets...')
    datasets = load_datasets(args, transforms)
    print(f"Loaded {len(datasets)} datasets")
    whole_train = merge_datasets(datasets, split = 'train')
    
    tokenizer, collator = prepare_tokenizer_and_collator(whole_train, args)
    train_dataloader = prepare_train_loaders(whole_train, collator, args.num_workers_train, args.batch_size)
    
    model = prepare_model(len(tokenizer), args)
    
    for batch in train_dataloader:
        print(model(batch).shape)
        break

if __name__ == '__main__': 
    
    args = parse_arguments()
    main(args)