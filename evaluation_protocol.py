import torch, torchvision
import wandb
import json

from src.io.args import parse_arguments, ListToArgsConstructor
from src.io.load_datasets import load_datasets
from src.vision.transfer_strategies import DataFixTransfer
from src.task_vectors import NonLinearTaskVector, LinearizedTaskVector
from src.linearize import LinearizedModel
from src.io.formatting_io_ops import preload_model

from main import prepare_model, evaluation_epoch, prepare_tokenizer_and_collator, merge_datasets

def prepare_datafix_model(model, collator, origin_dataset_list, target_dataset_list, transforms, args, eval_split = 'test'):
    # WARNING: It only works on encoder - decoder architectures as we have a feature space in the middle
    source_dataset_args = ListToArgsConstructor(origin_dataset_list, args)
    target_dataset_args = ListToArgsConstructor(target_dataset_list, args)

    test_split = load_datasets(target_dataset_args, transforms)
    source_dataset = merge_datasets(load_datasets(source_dataset_args, transforms), 'train')
    target_dataset = merge_datasets(test_split, eval_split)

    datafix_model = DataFixTransfer(model, collator, source_dataset, target_dataset,
                                    max_tokens = args.datafix_max_tokens)

    return datafix_model, test_split

def fuse_models(base_model, tokenizer_leng, args):

    vectorizer = LinearizedTaskVector if args.linear_model else NonLinearTaskVector
    # We need to linearize the model if it is not linear already
    if args.linear_model and not isinstance(base_model, LinearizedModel):
        base_model = LinearizedModel(base_model) # Taylor's first order expansion

    task_vectors = []
    weights = []
    for weight, model_checkpoint in zip(args.linear_sum_models_weights, args.checkpoints_list):

        if args.linear_model and not isinstance(base_model, LinearizedModel):
            print(f"(SCRIPT) Trying to operate with a non-linear FT version while --linear_model is true."
                  f"\n\tCheckpoint: {model_checkpoint}")
        loaded_model = prepare_model(tokenizer_leng, args)
        loaded_model.load_state_dict(torch.load(model_checkpoint))
        loaded_model.eval()

        # Important, everything linear or everything non-Linear
        task_vector = vectorizer(preload_model(base_model), preload_model(loaded_model)) # Like this?
        task_vectors.append(task_vector)
        weights.append(weight)

    multi_domain_vector = task_vectors[0]
    for prev_idx, task_vector in enumerate(task_vectors[1:]):
        multi_domain_vector = ((weights[prev_idx] if prev_idx == 0 else 1) * multi_domain_vector
                               + weights[prev_idx + 1] * task_vector)

    apply_method = multi_domain_vector.apply_to_linear if args.linear_model else multi_domain_vector.apply_to
    return apply_method(preload_model(base_model), args.final_vector_scaling).cuda()

def eval(args):
    results = {}

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
    if args.perform_model_arithmetics:
        model_vectorized = fuse_models(model, len(tokenizer), args)
    else: model_vectorized = None
    if args.perform_feature_correction:
        datafix_model, eval_datasets = prepare_datafix_model(model, collator, args.source_datasets,
                                                             args.target_datasets, transforms, args, 'test')
        print('------------------ Datafix - Dropout Evaluation Protocol -------------------------')

        for result in evaluation_epoch(eval_datasets, datafix_model, tokenizer, collator, args):
            print(result)

    else: eval_datasets = load_datasets(args, transforms)

    if args.perform_model_arithmetics:
        results['results_taylor'] = []

        print('------------------ Fused Models Evaluation Protocol -------------------------')
        assert model_vectorized is not None, 'How did you even get here?'
        for result in evaluation_epoch(eval_datasets, model_vectorized, tokenizer, collator, args):
            print(result)
            results['results_taylor'].append(result)

    results['results_baseline'] = []
    print('------------------ Common evaluation protocol -------------------------')
    for result in evaluation_epoch(eval_datasets, model, tokenizer, collator, args):
        print(result)
        results['results_baseline'].append(result)

    json.dump(results, open('results_output.json'))

if __name__ == '__main__':
    args = parse_arguments()
    eval(args)
