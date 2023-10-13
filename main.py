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
from src.evaluation.eval import eval_dataset

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

def evaluation_epoch(datasets, model, tokenizer, collator, args):
    
    for dataset in datasets:
        for split in ['val', 'test']:
            if dataset[split] is not None:
                dataset_name = f"{dataset[split].name}_{dataset[split].split}"
                print(f"Evaluation on {dataset_name} with {len(dataset[split])} samples")
                dataloader = torch.utils.data.DataLoader(dataset[split], batch_size = args.batch_size, collate_fn = collator.collate, num_workers = args.num_workers_test)
                
                eval_dataset(dataloader, model, dataset_name, tokenizer, wandb)
                
# We will heve to define training strategies for "simple", "continual" and "arithmetic".
## Arithmetic models are trained normally, but they are linearized with the modules

def loop(epoches, model, datasets, collator, tokenizer, args, train_function = None, **kwargs):
    
    ## TEMPORALLY LOOKING BAD. IT SHOULD RECEIVE PROPER ARGS NOT KWARGS ###
    for epoch in range(epoches):
        
        # train_function()
        print(f"{epoch} / {epoches} epoches")
        evals = evaluation_epoch(datasets, model, tokenizer, collator, args)
        print(evals)

def main(args):
    
    
    model_name = get_model_name(args)
    print(model_name)

    
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
    
    wandb.init(project='oda_ocr')
    wandb.config.update(args)
    wandb.run.name = model_name
    
    loop(args.epoches, model, datasets, collator, tokenizer, args)

if __name__ == '__main__': 
    
    args = parse_arguments()
    main(args)