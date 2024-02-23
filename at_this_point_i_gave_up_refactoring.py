import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

from src.io.args import parse_arguments, ListToArgsConstructor
from evaluation_protocol import fuse_two_models
from main import prepare_tokenizer_and_collator, prepare_model
from src.io.load_datasets import load_datasets, log_usage
from measure_distances_of_models import NORMS, p_norm
from main import evaluation_epoch
import copy

import numpy as np
import torchvision
import torch
import os
import wandb

def measure_model_distances(left_model_dict, right_model_dict, norm=2):

    v1, v2 = [], []
    for left_key, right_key in zip(left_model_dict, right_model_dict):
        assert left_key == right_key, ('Linearizing models is not implemented yet,'
                                       'compare equal things or implement it')

        v1.extend(left_model_dict[left_key]
                  .view(-1).cpu()
                  .numpy().tolist()
                  )
        v2.extend(right_model_dict[right_key]
                  .view(-1).cpu()
                  .numpy().tolist()
                  )
    if isinstance(norm, int):
        weighter = p_norm
    elif norm in NORMS:
        weighter = NORMS[norm]
    else:
        raise NotImplementedError(f"{norm} is not an implemented distance")
    return weighter(v1, v2, norm)

args = parse_arguments()

os.environ["WANDB_MODE"] = "disabled"
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

limit = 175
step = 15
weights = [(i/limit) for i in range(0, limit, step)]
if 1 not in weights:
    weights += [1]

model_a = torch.load(
    '/data/users/amolina/oda_ocr_output/non_linear_word_art_from_hiertext/non_linear_word_art_from_hiertext.pt')
model_b = torch.load(
    '/data/users/amolina/oda_ocr_output/non_linear_totaltext_from_hiertext/non_linear_totaltext_from_hiertext.pt')
model_both = torch.load('/data/users/amolina/oda_ocr_output/'
                        'non_linear_word_art_and_totaltext_from_base'
                        '/non_linear_word_art_and_totaltext_from_base.pt')

metric_a = 'CER_word_art_dataset_validation'
metric_b = 'CER_total_text_dataset_Test'

name_a = 'word_art'
name_b = 'totaltext'

color_a = '#d63131'
color_b = '#319fd6'
color_avg = '#404040'

split_langs = True
splits = ['val', 'test']
eval_datasets = load_datasets(args, transforms, split_langs=split_langs)

#print(model)
with torch.no_grad():

    model_a_performance = []
    model_b_performance = []
    model_distances = []

    base_model = prepare_model(len(tokenizer), args)  # Here we define the base model
    report = evaluation_epoch(eval_datasets, base_model, tokenizer, collator, args, splits)
    fig, ax = plt.subplots(figsize=(15, 7))

    whole_dict = {}
    for dict_ in report:
        whole_dict = {**whole_dict, **dict_}

    origin_model_dataset_a_performance = [whole_dict[metric_a]] * len(weights)
    origin_model_dataset_b_performance = [whole_dict[metric_b]] * len(weights)
    average = [.5 * i + .5 * j for i, j in zip(origin_model_dataset_a_performance, origin_model_dataset_b_performance)]

    ax.plot(weights, origin_model_dataset_a_performance, marker=8, linestyle='-', color=('%s' % color_a), label=f'Origin model ({name_a})', linewidth=.5)
    ax.plot(weights, origin_model_dataset_b_performance, marker=8, linestyle='-', color=('%s' % color_b), label=f'Origin model ({name_b})', linewidth=.5)
    ax.plot(weights, average, marker=8, linestyle='dashdot', color=('%s' % color_avg), label='Origin model (avg)', linewidth=.25)


    base_model.load_state_dict(model_both)
    report = evaluation_epoch(eval_datasets, base_model, tokenizer, collator, args, splits)

    whole_dict = {}
    for dict_ in report:
        whole_dict = {**whole_dict, **dict_}

    joint_dataset_model_a_performance = [whole_dict[metric_a]] * len(weights)
    joint_dataset_model_b_performance = [whole_dict[metric_b]] * len(weights)
    average = [.5 * i + .5 * j for i, j in zip(joint_dataset_model_a_performance, joint_dataset_model_b_performance)]

    ax.plot(weights, joint_dataset_model_a_performance, marker='x', linestyle='-', color=color_a, label=f'Joint model ({name_a})', linewidth=.5)
    ax.plot(weights, joint_dataset_model_b_performance, marker='x', linestyle='-', color=color_b, label=f'Joint model ({name_b})', linewidth=.5)
    ax.plot(weights, average, linestyle='dashdot', marker='x', color=color_avg, label='Joint model (avg)', linewidth=.25)



    for weight in weights:
        print(f"Comparing weights {weight} - {1-weight}")

        base_model = prepare_model(len(tokenizer), args)  # Here we define the base model
        base_model_copy = copy.deepcopy(base_model)
        model = fuse_two_models(base_model, model_a, model_b, weight, 1 - weight, 1)
        report = evaluation_epoch(eval_datasets, model, tokenizer, collator, args, splits)
        whole_dict = {}

        model_distances.append(measure_model_distances(base_model_copy.state_dict(), model.state_dict()))

        for dict_ in report:
            whole_dict = {**whole_dict, **dict_}

        model_a_performance.append(whole_dict[metric_a])
        model_b_performance.append(whole_dict[metric_b])

        model.cpu(), base_model.cpu()
        del model, base_model
    max_performing = max(*model_a_performance, *model_b_performance,
                         *joint_dataset_model_a_performance, *joint_dataset_model_b_performance,
                         *origin_model_dataset_b_performance, *origin_model_dataset_a_performance)

    ax.scatter(y=model_a_performance, x=weights, color=color_a, label=f'{name_a} error')
    ax.plot(weights, model_a_performance, color=color_a, linestyle='dashed')

    ax.scatter(y=model_b_performance, x= weights, c=color_b, label=f'{name_b} error')
    ax.plot(weights, model_b_performance, c=color_b, linestyle='dashed')

    avg = [.5 * i + .5 * j for i, j in zip(model_b_performance, model_a_performance)]

    ax.scatter(y=avg, x= weights, c=color_avg, label='average error')
    ax.plot(weights, avg,
            c=color_avg, linestyle='dashed', linewidth=.25)

    plt.title(f'Non-Linear: (1 - p) * {name_a} + p * {name_b}')

    ax.set_ylim(None, max_performing)


    x_fill = np.linspace(min(weights), max(weights), 10000)
    y_fill = np.linspace(-max_performing, max_performing, 10000)
    X_fill, Y_fill = np.meshgrid(x_fill, y_fill)
    Z_fill = np.interp(X_fill, weights, model_distances)
    gradient = plt.contourf(X_fill, Y_fill, Z_fill, cmap='viridis', alpha=0.2)
    cbar = plt.colorbar(gradient, label='distance to origin')

    #plt.grid()
    plt.legend(ncol=3)
    plt.xlabel('p - Vector scaling factor')
    plt.ylabel('Character Error Rate')
    plt.savefig(f'{name_a}_and_{name_b}.png', dpi=300, transparent=True)

print(report)