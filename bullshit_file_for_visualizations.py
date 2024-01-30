import torch, torchvision
import wandb
import json
import os

from src.io.args import parse_arguments, ListToArgsConstructor
from src.io.load_datasets import load_datasets
from src.vision.transfer_strategies import DataFixTransfer
from src.task_vectors import NonLinearTaskVector, LinearizedTaskVector
from src.linearize import LinearizedModel
from src.io.formatting_io_ops import preload_model
from src.evaluation.visutils import plot_radial
from src.io.models_dictionary import MODELS_LUT

from main import prepare_model, evaluation_epoch, prepare_tokenizer_and_collator, merge_datasets
from evaluation_protocol import fuse_models

'''
Help command bc im lazy:

    WANDB_MODE=disabled python bullshit_file_for_visualizations.py \
    --device cuda --use_esposalles --use_svt --use_iiit \
    --model_architecture vit_atienza --decoder_architecture transformer --tokenizer_name oda_giga_tokenizer \
    --load_checkpoint
'''

if __name__ == '__main__':
    output_tmp_folder = '../TMP_VIZ/'
    os.makedirs(output_tmp_folder, exist_ok=True)

    args = parse_arguments()
    models = MODELS_LUT
    results_per_dataset = {}
    results_per_domain = {}

    #### COMMON STAGE ####
    wandb.init(project='oda_ocr_evals')
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
    eval_datasets = load_datasets(args, split_langs=True, transforms=transforms)
    for idx in range(len(eval_datasets)):
        eval_datasets[idx]['train'] = None
    for approach_tag, model_checkpoint in zip(models, models.values()):
        args.checkpoint_name = model_checkpoint
        model = prepare_model(len(tokenizer), args)

        results_per_dataset[approach_tag] = {}

        with wandb.init(project='oda_ocr_evals', name=approach_tag) as run:
            wandb.config.update(args, allow_val_change=True)

            print('------------------ Common evaluation protocol -------------------------')
            for result in evaluation_epoch(eval_datasets, model, tokenizer, collator, args):
                results_per_dataset[approach_tag] = {**results_per_dataset[approach_tag], **result}
                print(results_per_dataset)

                # Log the "results" variable
                run.log({'results': results_per_dataset[approach_tag]})

    ## PREPARE DATA PER DATASET ##
    ### GET CER
    for metric in ['CER_', 'ED_']:
        names = list(results_per_dataset.keys())

        unraveled_results = [
            {dataset: resultat for dataset, resultat in
             zip(results_per_dataset[approach].keys(), results_per_dataset[approach].values())
             if metric in dataset} for approach in names
        ]

        numbers = [[unraveled_results[idx][dataset_name] for dataset_name in unraveled_results[0]]
                   for idx in range(len(unraveled_results))]
        labels = [dataset_name for dataset_name in unraveled_results[0]]

        plot_radial(labels, numbers, names, output=output_tmp_folder + f'radial_tmp_{metric}.png')

    ## PREPARE DATA PER DOMAIN ###
    handwritten_domain = set(['esposalles', 'washington', 'parzivall', 'iam'])
    document_domain = set(['funsd', 'xfund', 'hist_maps', 'sroie'])
    scene_domain = set(['mlt', 'totaltext', 'textocr', 'cocotext', 'svt', 'iiit'])
