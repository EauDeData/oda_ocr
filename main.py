import os

from src.io.args import parse_arguments
from src.io.load_datasets import load_datasets
from src.dataloaders.summed_dataloader import CollateFNs
from src.tokenizers.char_tokenizer import CharTokenizer

def merge_datasets(datasets, split = 'train'):
    
    data = datasets[0][split]
    
    for idx in range(1, len(datasets)):
        
        data = data + datasets[idx][split]
        
    return data
    
    

def prepare_tokenizer_and_collator(merged_dataset, args):
  
  tokenizer = CharTokenizer(merged_dataset, args.tokenizer_location, args.tokenizer_name, args.save_tokenizer)
  collator = CollateFNs(args.patch_width, args.image_height, tokenizer)
  
  return tokenizer, collator
    
def prepare_models():
    pass

# We will heve to define training strategies for "simple", "continual" and "arithmetic".
## Arithmetic models are trained normally, but they are linearized with the modules

def main(args):
    
    transforms = lambda x: x
    
    print('Loading all datasets...')
    datasets = load_datasets(args, transforms)
    print(f"Loaded {len(datasets)} datasets")
    if not os.path.exists(args.tokenizer_location + args.tokenizer_name + '.json'):
        
        whole_train = merge_datasets(datasets, split = 'train')
        print(f'Total train size: {len(whole_train)}')
    
    else: whole_train = None
    
    tokenizer, collator = prepare_tokenizer_and_collator(whole_train, args)
    
    print(tokenizer(list('hello')))

if __name__ == '__main__': 
    
    args = parse_arguments()
    main(args)