import copy

import torch, torchvision
import wandb
import json

from src.io.args import parse_arguments, ListToArgsConstructor
from src.io.load_datasets import load_datasets
# from src.vision.transfer_strategies import DataFixTransfer
DataFixTransfer = None
from src.task_vectors import NonLinearTaskVector, LinearizedTaskVector, GenericLinearVectorizer
from src.linearize import LinearizedModel
from src.io.formatting_io_ops import preload_model
from typing import *
from main import prepare_model, evaluation_epoch, prepare_tokenizer_and_collator, merge_datasets
from src.evaluation.eval import eval_dataset_democracy, eval_dataset_for_print_mask

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
def obtain_task_vector(base_model_state_dict, model_state_dict):

    state_dict = {}
    for left_key, right_key in zip(base_model_state_dict, model_state_dict):
        assert left_key == right_key, ('Linearizing models is not implemented yet,'
                                       'compare equal things or implement it')
        state_dict[left_key] = base_model_state_dict[left_key] - model_state_dict[right_key]

    return state_dict

def apply_task_vector(model_to_be_applied, task_vector_state_dict, weight = 1):
    state_dict = {}
    for left_key, right_key in zip(model_to_be_applied, task_vector_state_dict):
        assert left_key == right_key, ('Linearizing models is not implemented yet,'
                                       'compare equal things or implement it')
        state_dict[left_key] = model_to_be_applied[left_key] + weight * task_vector_state_dict[right_key]

    return state_dict
def fuse_models(base_model, tokenizer_leng, args):

    # We need to linearize the model if it is not linear already
    if args.linear_model and not isinstance(base_model, LinearizedModel):
        base_model = LinearizedModel(base_model) # Taylor's first order expansion

    task_vectors = []
    weights = []
    model_to_mess_with = copy.deepcopy(base_model)
    for weight, model_checkpoint in zip(args.linear_sum_models_weights, args.checkpoints_list):

        print('Fusing checkpoint', model_checkpoint, 'with weight', weight)
        # Important, everything linear or everything non-Linear
        model_to_mess_with.load_state_dict(torch.load(model_checkpoint))
        task_vectors.append(obtain_task_vector(base_model.encoder.state_dict(),
                                               model_to_mess_with.encoder.state_dict()))
        weights.append(weight)

    multi_domain_vector = base_model.encoder.state_dict()

    for idx, (w, task_vector) in enumerate(zip(weights, task_vectors)):
        multi_domain_vector = apply_task_vector(multi_domain_vector, task_vector, w)

    base_model.encoder.load_state_dict(multi_domain_vector)
    return base_model.state_dict() # TODO: Care, using only encoder!!!



def fuse_two_models(base_model, model_a_state_dict, model_b_state_dict, weight_a, weight_b, final_scaling = 1):

    task_vector_a = GenericLinearVectorizer(base_model.state_dict(), model_a_state_dict) * weight_a
    task_vector_b = GenericLinearVectorizer(base_model.state_dict(), model_b_state_dict) * weight_b

    multi_vector = task_vector_a + task_vector_b

    return multi_vector.apply_to(base_model, final_scaling).cuda()

def get_list_of_checkpoints(args):
    return [torch.load(ckpt) for ckpt in args.checkpoints_list]

class DemocracyMaker9999(torch.nn.Module):
    def __init__(self, model: torch.nn.Module, checkpoints: List[dict], device):
        super().__init__()
        self.ckpoints = checkpoints
        self.model = model
        self.softmax = torch.nn.Softmax(2) # (0_seq, 1_batch, 2_emb)
        self.to(device)
    def __call__(self, batch, **kwargs):
        predicted_sequences = []
        for state_dict in self.ckpoints:
            self.model.load_state_dict(state_dict)
            prediction = self.softmax(self.model(batch)['language_head_output'])
            predicted_sequences.append(prediction)

        return {'language_head_output': sum(predicted_sequences)}


'''
Exmaple command: WANDB_MODE=disabled python evaluation_protocol.py
--use_iam --device cpu --use_mlt --model_architecture vit_atienza
--decoder_architecture transformer
--tokenizer_name oda_giga_tokenizer
--batch_size 5 --checkpoints_list [CKPTS HERE] --do_democracy

'''

def eval(args):
    results = {}

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

    if args.do_democracy:
        print('------------------ Voting Ensemble Evaluation Protocol -------------------------')
        democratic_movement_of_weights = DemocracyMaker9999(
            model, get_list_of_checkpoints(args), args.device
        )
        results['results_voting'] = []
        for result in evaluation_epoch(eval_datasets, democratic_movement_of_weights, tokenizer, collator, args):
            print(result)
            results['results_voting'].append(result)

    if args.do_neuron_inspection:
        print('------------------ Neuron inspection Evaluation Protocol -------------------------')

        for result in evaluation_epoch(eval_datasets, model, tokenizer, collator, args, splits=['val', 'test'],
                                       eval_fn=eval_dataset_for_print_mask):
            print(result)
        print('Neuron inspection ends the process, dying...')
        exit()

    results['results_baseline'] = []
    print('------------------ Common evaluation protocol -------------------------')
    for result in evaluation_epoch(eval_datasets, model, tokenizer, collator, args, splits=['val', 'test'],
                                   ):
        print(result)
        results['results_baseline'].append(result)

    json.dump(results, open('results_output.json'))

if __name__ == '__main__':
    args = parse_arguments()
    eval(args)
