import os
import torch, torchvision
import numpy as np
import wandb
import torch.optim as optim
from vit_pytorch import ViT
import uuid

from src.io.args import parse_arguments, get_model_name, model_choices_lookup
from src.io.load_datasets import load_datasets
from src.dataloaders.summed_dataloader import CollateFNs
from src.tokenizers.char_tokenizer import CharTokenizer
from src.vision.models import (ViTEncoder, ConvVitEncoder, _ProtoModel, CLIPWrapper, RNNDecoder, ViTAtienzaWrapper,
                               TransformerDecoder)
from src.vision.vitstr import vitstr_base_patch16_224
from src.linearize import LinearizedModel, AllMightyWrapper
from src.evaluation.eval import eval_dataset
from src.evaluation.visutils import loop_for_visualization
from src.train_steps.base_ctc import train_ctc, train_ctc_clip
from src.train_steps.base_cross_entropy import train_cross_entropy

from main import prepare_model, evaluation_epoch, prepare_tokenizer_and_collator, merge_datasets

def eval(args):
    
    normalize = {
        'normalize': lambda x: (x - x.min()) / max((x.max() - x.min()), 0.01),
        'standarize': lambda x: x / max(x.max(), 1)
    }

    transforms = torchvision.transforms.Compose((
        torchvision.transforms.PILToTensor(),
        normalize['normalize' if not args.standarize else 'standarize'])
    )
    
    datasets = load_datasets(args, transforms)
    whole_train = merge_datasets(datasets, split='train')
    tokenizer, collator = prepare_tokenizer_and_collator(whole_train, transforms, args)

    model = prepare_model(len(tokenizer), args)

    for result in evaluation_epoch(datasets, model, tokenizer, collator, args):
        print(result)

if __name__ == '__main__':
    args = parse_arguments()
    eval(args)
