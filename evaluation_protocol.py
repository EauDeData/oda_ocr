import torch, torchvision
import wandb

from src.io.args import parse_arguments, ListToArgsConstructor
from src.io.load_datasets import load_datasets
from src.vision.transfer_strategies import DataFixTransfer

from main import prepare_model, evaluation_epoch, prepare_tokenizer_and_collator, merge_datasets

def prepare_datafix_model(model, collator, origin_dataset_list, target_dataset_list, transforms, args, eval_split = 'test'):
    # WARNING: It only works on encoder - decoder architectures as we have a feature space in the middle
    source_dataset_args = ListToArgsConstructor(origin_dataset_list, args)
    target_dataset_args = ListToArgsConstructor(target_dataset_list, args)

    source_dataset = merge_datasets(load_datasets(source_dataset_args, transforms), 'train')
    target_dataset = merge_datasets(load_datasets(target_dataset_args, transforms), eval_split)

    datafix_model = DataFixTransfer(model, collator, source_dataset, target_dataset,
                                    max_tokens = args.datafix_max_tokens)

    return datafix_model, target_dataset

def eval(args):

    wandb.init(project='oda_ocr')
    wandb.config.update(args)
    wandb.run.name = f"evaluation_{args.checkpoint_name}" + '_datafix' if args.perform_feature_correction else ''

    normalize = {
        'normalize': lambda x: (x - x.min()) / max((x.max() - x.min()), 0.01),
        'standarize': lambda x: x / max(x.max(), 1)
    }

    transforms = torchvision.transforms.Compose((
        torchvision.transforms.PILToTensor(),
        normalize['normalize' if not args.standarize else 'standarize'])
    )
    

    tokenizer, collator = prepare_tokenizer_and_collator(None, transforms, args)

    model = prepare_model(len(tokenizer), args)
    if args.perform_feature_correction:
        datafix_model, eval_datasets = prepare_datafix_model(model, collator, args.source_datasets,
                                                             args.target_datasets, transforms, args, 'test')
        print('------------------ Datafix - Dropout Evaluation Protocol -------------------------')

        for result in evaluation_epoch(eval_datasets, datafix_model, tokenizer, collator, args):
            print(result)

    else: eval_datasets = load_datasets(args, transforms)

    print('------------------ Common evaluation protocol -------------------------')
    for result in evaluation_epoch(eval_datasets, model, tokenizer, collator, args):
        print(result)

if __name__ == '__main__':
    args = parse_arguments()
    eval(args)
