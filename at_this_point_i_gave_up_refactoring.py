import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

from src.io.args import parse_arguments, ListToArgsConstructor
from evaluation_protocol import fuse_two_models
from main import prepare_tokenizer_and_collator, prepare_model
from src.io.load_datasets import load_datasets, log_usage
from measure_distances_of_models import NORMS, p_norm
from main import evaluation_epoch
import copy
from src.task_vectors_original import TaskVector
from measure_distances_of_models import from_vector_to_state_dict
import numpy as np
import torchvision
import torch
import os
import wandb
from tqdm import tqdm

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


def compute_weighted_average(base_checkpoint, list_of_checkpoints, list_of_weights):
    assert len(list_of_checkpoints) == len(list_of_weights), "The length of checkpoints and weights must match."

    models = []
    tv1 = None

    for num_k, base_model in tqdm(enumerate(list_of_checkpoints), total=len(list_of_checkpoints)):
        v1 = []
        tv1 = TaskVector(base_checkpoint, base_model).vector
        for left_key in tv1:
            v1.extend(tv1[left_key]
                      .reshape(torch.numel(tv1[left_key])).cpu()
                      .numpy().tolist()
                      )
        t = torch.tensor(v1, device='cuda')
        models.append(t * list_of_weights[num_k])

    models = torch.stack(models)
    weighted_mean_model = models.sum(dim=0)

    mean_model_state_dict = TaskVector(vector=from_vector_to_state_dict(tv1, weighted_mean_model)) \
        .apply_to(base_checkpoint)

    return mean_model_state_dict

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

model_a = args.model_a_for_dist # '/data2/users/amolina/oda_ocr_output/langs_domain_adaptation/few_shot_chinese_from_averaged_from_hiertext/few_shot_chinese_from_averaged_from_hiertext.pt'
model_b = args.model_b_for_dist # '/data2/users/amolina/oda_ocr_output/langs_domain_adaptation/few_shot_korean_from_averaged_from_hiertext/few_shot_korean_from_averaged_from_hiertext.pt'
model_both = torch.load(args.model_joint_both)

base_checkpoint = args.model_joint_both # '/data2/users/amolina/oda_ocr_output/svd/from_hiertext/averaged_model.pth'
metric_a = args.metric_a_for_dist #  'CER_mlt19_dataset_Chi_val_cv1'
metric_b = args.metric_b_for_dist # 'CER_mlt19_dataset_Kor_val_cv1'

name_a = args.name_a_for_dist # 'Chinese'
name_b = args.name_b_for_dist # 'Korean'

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

    # ax.plot(weights, origin_model_dataset_a_performance, marker=8, linestyle='-', color=('%s' % color_a), label=f'Origin model ({name_a})', linewidth=.5)
    # ax.plot(weights, origin_model_dataset_b_performance, marker=8, linestyle='-', color=('%s' % color_b), label=f'Origin model ({name_b})', linewidth=.5)
    # ax.plot(weights, average, marker=8, linestyle='dashdot', color=('%s' % color_avg), label='Origin model (avg)', linewidth=.25)


    base_model.load_state_dict(model_both)
    report = evaluation_epoch(eval_datasets, base_model, tokenizer, collator, args, splits)

    whole_dict = {}
    for dict_ in report:
        whole_dict = {**whole_dict, **dict_}

    joint_dataset_model_a_performance = [whole_dict[metric_a]] * len(weights)
    joint_dataset_model_b_performance = [whole_dict[metric_b]] * len(weights)
    average = [.5 * i + .5 * j for i, j in zip(joint_dataset_model_a_performance, joint_dataset_model_b_performance)]

    # ax.plot(weights, joint_dataset_model_a_performance, marker='x', linestyle='-', color=color_a, label=f'Joint model ({name_a})', linewidth=.5)
    # ax.plot(weights, joint_dataset_model_b_performance, marker='x', linestyle='-', color=color_b, label=f'Joint model ({name_b})', linewidth=.5)
    # ax.plot(weights, average, linestyle='dashdot', marker='x', color=color_avg, label='Joint model (avg)', linewidth=.25)



    for weight in weights:
        print(f"Comparing weights {1- weight, name_a} - {weight, name_b}")

        base_model = prepare_model(len(tokenizer), args)  # Here we define the base model
        model = copy.deepcopy(base_model)
        model_state_dict = compute_weighted_average(base_checkpoint, [model_a, model_b], [1-weight, weight])
        model.load_state_dict(model_state_dict)
        report = evaluation_epoch(eval_datasets, model, tokenizer, collator, args, splits)
        whole_dict = {}

        model_distances.append(measure_model_distances(base_model.state_dict(), model.state_dict()))

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
    plt.legend(ncol=1)
    plt.xlabel('p - Vector scaling factor')
    plt.ylabel('Character Error Rate')
    plt.savefig(f'{name_a}_and_{name_b}.svg', dpi=300, transparent=True)

print(report)
