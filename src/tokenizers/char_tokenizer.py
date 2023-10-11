import os
import json
from tqdm import tqdm
import math
import numpy as np

class CharTokenizer:

    '''
        This tokenizer may be inputted in our collate FN class so we can put it on the dataloader.
            It's elegant (I think)
    
    '''

    bos = '<BOS>'
    eos = '<EOS>'
    unk = '<UNK>'
    cls_token = '<CLS>'

    def __init__(self, dataset = None, local_path = 'tmp_/tokenizers/', tokenizer_name = 'tokenizer', save_on_init = True) -> None:
        
        os.makedirs(local_path, exist_ok=True)
        self.full_path = os.path.join(local_path, tokenizer_name + '.json')
        if os.path.exists(
            self.full_path
        ):
            print(f'Tokenizer {tokenizer_name} found in {local_path}, loading tokens from local storage.')
            self.tokens = json.load(
                open(self.full_path, 'r')
            )

            return None
        
        self.init_tokens(dataset, save_on_init)

    def __len__(self):
        return len(self.tokens)
    
    def __call__(self, tokens: list) -> np.ndarray:

        return np.array([
            self.tokens[token] if token in self.tokens else self.tokens[self.unk]
            
                for token in [self.bos, self.cls_token] + tokens + [self.eos]
        ])

    
    def init_tokens(self, dataset, save):

        tokens_with_freqs = {
            self.bos: math.inf,
            self.eos: math.inf,
            self.cls_token: math.inf,
            self.unk: 0
            }
        
        for idx in tqdm(range(len(dataset)), desc='tokenizing dataset...'):
            
            for char in dataset[idx]['tokens']:

                if not char in tokens_with_freqs: tokens_with_freqs[char] = 0
                tokens_with_freqs[char] += 1

        self.tokens = {token: num for num, token in enumerate(sorted(tokens_with_freqs.keys(), reverse = True, key = lambda x: tokens_with_freqs[x]))}
        if save:
            print(f"Tokens saved at {self.full_path}!")
            json.dump(
                self.tokens, open(self.full_path, 'w')
            ) 

