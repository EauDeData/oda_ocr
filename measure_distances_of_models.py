import wandb
import os
import torch
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import math
import cv2
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
from src.task_vectors_original import TaskVector
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from src.io.args import parse_arguments
from src.io.models_dictionary import MODELS_LUT
from scipy.spatial.distance import cosine
from scipy.stats import wasserstein_distance
def p_norm(v1, v2, p, reduce_fn=sum):
    return reduce_fn([abs(i - j) ** p for i, j in zip(v1, v2)])**(1/p)
def cosine_ignorer(v1, v2, *args):
    return 1 - cosine(v1, v2)
def divergence_ignorer(v1, v2, *args):
    return wasserstein_distance(v1, v2)

NORMS = {'cosine': cosine_ignorer, 'divergence': divergence_ignorer}

def compute_model_distances(left_model_dict, right_model_dict, norm = 2):

    v1 = []
    v2 = []

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

    square_size = int(math.sqrt(len(v1)))

    return weighter(v1, v2, 1, lambda x: np.array(x)), v1, v2, square_size, weighter
def model_distance(models_lut, filepath, norm = 2):

    graph = nx.Graph()
    done_models = []
    for base_model in models_lut:
        for model_checkpoint in [model for model in models_lut if model!=base_model]:
            if (base_model, model_checkpoint) in done_models: continue

            left_model_dict, right_model_dict = torch.load(models_lut[model_checkpoint]),\
                torch.load(models_lut[base_model])

            differences, v1, v2, square_size, weighter = compute_model_distances(left_model_dict, right_model_dict)

            differences = np.reshape(differences[:square_size ** 2],
                             (square_size, square_size))
            img = 255 * (differences - differences.min())/(differences.max() - differences.min())
            cv2.imwrite(f'tmp_/{model_checkpoint}_{base_model}_weights_diff.png', img.astype(np.uint8),
                        [cv2.IMWRITE_PNG_COMPRESSION, 0]
                        )

            graph.add_edge(base_model, model_checkpoint, weight=weighter(v1, v2, norm))
            done_models.append((model_checkpoint, base_model))

    nx.draw(graph, nx.spring_layout(graph), with_labels=True, node_size=700, node_color="skyblue", font_size=10)
    nx.draw_networkx_edge_labels(graph, nx.spring_layout(graph), edge_labels=\
        {(n1, n2): f"{d['weight']:.2f}" for n1, n2, d in graph.edges(data=True)})

    # Save the graph to a file
    plt.savefig(filepath, format="PNG")
    print({(n1, n2): f"{d['weight']:.2f}" for n1, n2, d in graph.edges(data=True)})
    nx.write_gexf(graph, filepath.replace('.png', '.gexf'))

def compute_cumulative_top_neurons_change(models_lut, model_to_rate_change):

    to_compare_dictionary = torch.load(models_lut[model_to_rate_change])
    done_models = []
    for base_model in models_lut:
        for model_checkpoint in [model for model in models_lut if model!=base_model]:
            if (base_model, model_checkpoint) in done_models: continue
            print(f"FT on {base_model} & {model_checkpoint}")

            left_model_dict, right_model_dict = torch.load(models_lut[model_checkpoint]),\
                torch.load(models_lut[base_model])
            differences_1, v1, v2, square_size, weighter =\
                compute_model_distances(left_model_dict, to_compare_dictionary, norm=1)

            differences_2, v1, v2, square_size, weighter =\
                compute_model_distances(right_model_dict, to_compare_dictionary, norm=1)

            argsorted_diff1_list = torch.tensor(differences_1.argsort(), device='cuda')
            argsorted_diff2_list = torch.tensor(differences_2.argsort(), device='cuda')

            total_hists = abs(argsorted_diff1_list - argsorted_diff2_list).detach().cpu().numpy()
            differences = np.reshape(total_hists[:square_size ** 2],
                             (square_size, square_size))
            img = 255 * (differences - differences.min())/(differences.max() - differences.min())
            cv2.imwrite(f'tmp_/{model_checkpoint}_{base_model}_weights_diff.png', img.astype(np.uint8),
                        [cv2.IMWRITE_PNG_COMPRESSION, 0]
                        )

            plt.scatter(total_hists[50:], range(50, len(total_hists)),
                     label=f"FT on {base_model} & {model_checkpoint}")
            plt.savefig(f'./tmp_/{base_model}_{model_checkpoint}_cumulative_weight_change.png')
            plt.clf()
            done_models.append((model_checkpoint, base_model))


def task_vectors_cosine_similarity_matrix(base_checkpoint, list_of_checkpoints):

    done_models = []
    matrix = np.ones([len(list_of_checkpoints)] * 2)
    for num_k, base_model in enumerate(list_of_checkpoints):
        for num_j, model_checkpoint in enumerate(list_of_checkpoints):
            if (base_model, model_checkpoint) in done_models or (base_model==model_checkpoint): continue
            done_models.append((model_checkpoint, base_model))
            print(f"FT on {base_model} & {model_checkpoint}")

            tv1 = TaskVector(base_checkpoint, base_model).vector
            tv2 = TaskVector(base_checkpoint, model_checkpoint).vector
            differences_1, v1, v2, square_size, weighter =\
                compute_model_distances(tv1, tv2, norm='cosine')
            matrix[num_j, num_k] = matrix[num_k, num_j] = differences_1

    return matrix

if __name__ == '__main__':
    output_tmp_folder = '../TMP_VIZ/'
    os.makedirs(output_tmp_folder, exist_ok=True)

    args = parse_arguments()
    models = MODELS_LUT
    results_per_dataset = {}
    results_per_domain = {}

    #### COMMON STAGE ####
    #model_distance(models, os.path.join(output_tmp_folder, 'model_distances.png'), 2)
    #compute_cumulative_top_neurons_change(models, 'Zero-Shot')
    base_model = models.pop('Zero-Shot (hiertext)')
    matrix = task_vectors_cosine_similarity_matrix(base_model, models.values())
    df_distance = pd.DataFrame(matrix, index=list(models.keys()), columns=list(models.keys()))

    # Create a heatmap using Seaborn
    plt.figure(figsize=(8*3, 6*3))
    hm = sns.heatmap(df_distance, annot=True, cmap='viridis', fmt='.2f', linewidths=.5)
    hm.set_yticklabels(hm.get_yticklabels(), rotation=45, horizontalalignment='right')
    plt.title('Model Distance Matrix')
    plt.xlabel('Models')
    plt.ylabel('Models')
    plt.savefig('../TMP_VIZ/cosine_distance.png')