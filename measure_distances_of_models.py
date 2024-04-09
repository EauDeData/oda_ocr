import copy

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
from src.ties_merging_minimal import ties_merging, vector_to_state_dict, state_dict_to_vector

from scipy.spatial.distance import cosine
from scipy.stats import wasserstein_distance


def p_norm(v1, v2, p, reduce_fn=sum):
    return reduce_fn([abs(i - j) ** p for i, j in zip(v1, v2)]) ** (1 / p)


def cosine_ignorer(v1, v2, *args):
    return 1 - cosine(v1, v2)


def divergence_ignorer(v1, v2, *args):
    return wasserstein_distance(v1, v2)


def histogram_intersection(v1, v2, *args):
    tensor = torch.stack((
        torch.tensor(v1, device='cuda', dtype=torch.float64), torch.tensor(v2, device='cuda', dtype=torch.float64))
    )
    minimum, _ = torch.min(tensor, dim=1, keepdim=True)
    tensor = (tensor - minimum)
    tensor = tensor / torch.sum(tensor, dim=1, keepdim=True)

    return torch.min(tensor, dim=0)[0].sum().cpu().detach().item()


NORMS = {'cosine': cosine_ignorer, 'divergence': divergence_ignorer, 'hist': histogram_intersection}


def compute_model_distances(left_model_dict, right_model_dict, norm=2):
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


def model_distance(models_lut, filepath, norm=2):
    graph = nx.Graph()
    done_models = []
    for base_model in models_lut:
        for model_checkpoint in [model for model in models_lut if model != base_model]:
            if (base_model, model_checkpoint) in done_models: continue

            left_model_dict, right_model_dict = torch.load(models_lut[model_checkpoint]), \
                torch.load(models_lut[base_model])

            differences, v1, v2, square_size, weighter = compute_model_distances(left_model_dict, right_model_dict)

            differences = np.reshape(differences[:square_size ** 2],
                                     (square_size, square_size))
            img = 255 * (differences - differences.min()) / (differences.max() - differences.min())
            cv2.imwrite(f'tmp_/{model_checkpoint}_{base_model}_weights_diff.png', img.astype(np.uint8),
                        [cv2.IMWRITE_PNG_COMPRESSION, 0]
                        )

            graph.add_edge(base_model, model_checkpoint, weight=weighter(v1, v2, norm))
            done_models.append((model_checkpoint, base_model))

    nx.draw(graph, nx.spring_layout(graph), with_labels=True, node_size=700, node_color="skyblue", font_size=10)
    nx.draw_networkx_edge_labels(graph, nx.spring_layout(graph), edge_labels= \
        {(n1, n2): f"{d['weight']:.2f}" for n1, n2, d in graph.edges(data=True)})

    # Save the graph to a file
    plt.savefig(filepath, format="PNG")
    print({(n1, n2): f"{d['weight']:.2f}" for n1, n2, d in graph.edges(data=True)})
    nx.write_gexf(graph, filepath.replace('.png', '.gexf'))


def compute_cumulative_top_neurons_change(models_lut, model_to_rate_change):
    to_compare_dictionary = torch.load(models_lut[model_to_rate_change])
    done_models = []
    for base_model in models_lut:
        for model_checkpoint in [model for model in models_lut if model != base_model]:
            if (base_model, model_checkpoint) in done_models: continue
            print(f"FT on {base_model} & {model_checkpoint}")

            left_model_dict, right_model_dict = torch.load(models_lut[model_checkpoint]), \
                torch.load(models_lut[base_model])
            differences_1, v1, v2, square_size, weighter = \
                compute_model_distances(left_model_dict, to_compare_dictionary, norm=1)

            differences_2, v1, v2, square_size, weighter = \
                compute_model_distances(right_model_dict, to_compare_dictionary, norm=1)

            argsorted_diff1_list = torch.tensor(differences_1.argsort(), device='cuda')
            argsorted_diff2_list = torch.tensor(differences_2.argsort(), device='cuda')

            total_hists = abs(argsorted_diff1_list - argsorted_diff2_list).detach().cpu().numpy()
            differences = np.reshape(total_hists[:square_size ** 2],
                                     (square_size, square_size))
            img = 255 * (differences - differences.min()) / (differences.max() - differences.min())
            cv2.imwrite(f'tmp_/{model_checkpoint}_{base_model}_weights_diff.png', img.astype(np.uint8),
                        [cv2.IMWRITE_PNG_COMPRESSION, 0]
                        )

            plt.scatter(total_hists[50:], range(50, len(total_hists)),
                        label=f"FT on {base_model} & {model_checkpoint}")
            plt.savefig(f'./tmp_/{base_model}_{model_checkpoint}_cumulative_weight_change.png')
            plt.clf()
            done_models.append((model_checkpoint, base_model))


def task_vectors_cosine_similarity_matrix(base_checkpoint, list_of_checkpoints, dist='hist'):
    done_models = []
    matrix = np.ones([len(list_of_checkpoints)] * 2)
    for num_k, base_model in enumerate(list_of_checkpoints):
        for num_j, model_checkpoint in enumerate(list_of_checkpoints):
            if (base_model, model_checkpoint) in done_models or (base_model == model_checkpoint): continue
            done_models.append((model_checkpoint, base_model))
            print(f"FT on {base_model} & {model_checkpoint}")

            tv1 = TaskVector(base_checkpoint, base_model).vector
            tv2 = TaskVector(base_checkpoint, model_checkpoint).vector
            differences_1, v1, v2, square_size, weighter = \
                compute_model_distances(tv1, tv2, norm=dist)
            matrix[num_j, num_k] = matrix[num_k, num_j] = differences_1  # This is similarity

    df_distance = pd.DataFrame(matrix, index=list_of_checkpoints, columns=list_of_checkpoints)

    # Create a heatmap using Seaborn
    plt.figure(figsize=(8 * 2, 6 * 2))
    hm = sns.heatmap(df_distance, annot=True, cmap='viridis', fmt='.2f', linewidths=.5)
    hm.set_yticklabels(hm.get_yticklabels(), rotation=45, horizontalalignment='right')
    plt.title('Model Distance Matrix')
    plt.xlabel('Models')
    plt.ylabel('Models')
    plt.savefig('../TMP_VIZ/cosine_distance.png')
    return matrix


def plot_values_task_vectors(base_checkpoint, list_of_checkpoints):
    for num_k, base_model in tqdm(enumerate(list_of_checkpoints), total=len(list_of_checkpoints)):

        v1 = []
        tv1 = TaskVector(base_checkpoint, base_model).vector
        for left_key in tv1:
            v1.extend(tv1[left_key]
                      .view(-1).cpu()
                      .numpy().tolist()
                      )
        square_size = int(math.sqrt(len(v1)))
        v1_vect = np.array(v1)[:square_size ** 2].reshape((square_size, square_size))
        normed_vect = (v1_vect - v1_vect.mean()) / v1_vect.std()

        standarized = 255 * (normed_vect - normed_vect.min()) / (normed_vect.max() - normed_vect.min())
        print('../TMP_VIZ/' + str(num_k) + '_task_vector.png')
        cv2.imwrite('../TMP_VIZ/' + str(num_k) + '_task_vector.png', standarized.astype(np.uint8))


def from_vector_to_state_dict(state_dict_base, V):
    state_dict_out = {key: None for key in state_dict_base}
    V = V.view(-1)
    print('Fitting vector of size', V.shape)
    # Iterate over the keys in the state_dict and update them with values from the vector
    index = 0
    for key, value in state_dict_base.items():
        # Get the number of elements in the tensor corresponding to the key
        num_elements = value.numel()

        # Extract a slice of appropriate size from the vector V
        slice_V = V[index:index + num_elements]

        # Reshape the slice to match the shape of the tensor in state_dict
        slice_V = slice_V.view(*value.shape)

        # Update the tensor in the state_dict with the values from slice_V
        state_dict_out[key] = slice_V

        # Move the index pointer forward
        index += num_elements
    return state_dict_out


def compute_svd(base_checkpoint, list_of_checkpoints):
    models = []
    tv1 = None
    print(list(list_of_checkpoints))
    for num_k, base_model in tqdm(enumerate(list_of_checkpoints), total=len(list_of_checkpoints)):

        v1 = []
        tv1 = TaskVector(base_checkpoint, base_model).vector
        for left_key in tv1:
            v1.extend(tv1[left_key]
                      .reshape(torch.numel(tv1[left_key])).cpu()
                      .numpy().tolist()
                      )
        t = torch.tensor(v1, device='cuda')
        models.append(t)
    models = torch.stack(models)
    u, s, v = torch.pca_lowrank(models)
    # most explicative components and see if its more generic
    print('svd computed:')
    print(u.shape, s.shape, v.shape)
    print('Eigenvalues', s)
    mean_model = models.mean(0)  # TODO: From here create a model which is from the i-th\

    mean_model_state_dict = TaskVector(vector=from_vector_to_state_dict(tv1, mean_model)) \
        .apply_to(base_checkpoint)

    base = '/data2/users/amolina/oda_ocr_output/svd/from_hiertext_hwonly/'
    os.makedirs(base, exist_ok=True)
    torch.save(mean_model_state_dict, base + 'averaged_model.pth')

    scaled_eigenvectors = torch.matmul(torch.diag(s), v.T)
    for i in range(v.shape[1]):
        # We save eigenvectors and scaled eigenvectors
        tvector = TaskVector(vector=from_vector_to_state_dict(tv1, v[:, i]))
        torch.save(tvector.apply_to(base_checkpoint, scaling_coef=1.0), base + f'{i}_eigenvector.pth')

        tvector = TaskVector(vector=from_vector_to_state_dict(tv1, scaled_eigenvectors[i]))
        torch.save(tvector.apply_to(base_checkpoint, scaling_coef=1.0), base + f'{i}_eigenvector_scaled.pth')

    for k in range(1, v.shape[1] + 1):
        projection = torch.mm(torch.mm(u[:, :k], torch.diag(s[:k])), v[:, :k].t())
        tvector = TaskVector(vector=from_vector_to_state_dict(tv1, projection.mean(0)))

        mean_model_state_dict = tvector.apply_to(base_checkpoint, scaling_coef=1.0)
        torch.save(mean_model_state_dict, base + f'_pca_{k}_average.pth')

def compute_average_scaled(base_checkpoint, list_of_checkpoints):
    models = []
    tv1 = None
    print(list(list_of_checkpoints))
    for num_k, base_model in tqdm(enumerate(list_of_checkpoints), total=len(list_of_checkpoints)):

        v1 = []
        tv1 = TaskVector(base_checkpoint, base_model).vector
        for left_key in tv1:
            v1.extend(tv1[left_key]
                      .reshape(torch.numel(tv1[left_key])).cpu()
                      .numpy().tolist()
                      )
        t = torch.tensor(v1, device='cuda')
        models.append(t)
    models = torch.stack(models)
    u, s, v = torch.pca_lowrank(models)
    # most explicative components and see if its more generic
    print('svd computed:')
    print(u.shape, s.shape, v.shape)
    print('Eigenvalues', s)
    mean_model = models.mean(0)  # TODO: From here create a model which is from the i-th\

    mean_model_state_dict = TaskVector(vector=from_vector_to_state_dict(tv1, mean_model)) \
        .apply_to(base_checkpoint)

    base = '/data2/users/amolina/oda_ocr_output/svd/from_hiertext_hwonly/'
    os.makedirs(base, exist_ok=True)
    torch.save(mean_model_state_dict, base + 'averaged_model.pth')


def fill_under_lines(ax=None, alpha=.1, **kwargs):
    if ax is None:
        ax = plt.gca()
    for line in ax.lines:
        x, y = line.get_xydata().T
        ax.fill_between(x, 0, y, color=line.get_color(), alpha=alpha, **kwargs)


def plot_histograms(dictionary):
    x = [i[1] for i in dictionary.values()]
    y = [i[0] for i in dictionary.values()]

    # Create a DataFrame
    data = {'x': [], 'y': [], 'label': []}
    for i, label in enumerate(dictionary):
        for x_i, y_i in zip(x[i], y[i]):
            data['x'].append(x_i)
            data['y'].append(y_i)
            data['label'].append(label)
    df = pd.DataFrame(data)
    # Plot
    plt.figure(figsize=(10, 6))
    sns.set_style("darkgrid")
    ax = sns.lineplot(data=df, x='x', y='y', hue='label')
    fill_under_lines(ax=ax)
    plt.xlabel('Weight difference')
    plt.ylabel('Frequency')
    plt.xlim(-0.05, 0.05)
    plt.ylim(0, 400)
    plt.title('Histograms of Vectors')
    plt.legend()
    plt.savefig('../TMP_VIZ/histogram_of_task_vectors.png', dpi=300, transparent=True)


def prepare_dictionary_of_vectors(models_lut, base_model_ckpt_pth):
    hists = {}
    for name, checkpoint in tqdm(models_lut.items()):

        v1 = []
        tv1 = TaskVector(base_model_ckpt_pth, checkpoint).vector
        for left_key in tv1:
            v1.extend(tv1[left_key]
                      .view(-1).cpu()
                      .numpy().tolist()
                      )
        hists[name] = np.histogram(np.array(v1), bins=100, density=True)

    print('plotting histograms...')
    plot_histograms(hists)


def task_vectors_cosine_similarity_matrix_eigen(base_checkpoint, list_of_checkpoints, list_of_eigencheckpoints,
                                                checkpoints_names=None):
    matrix = np.ones([len(list_of_checkpoints), len(list_of_eigencheckpoints)])
    for num_k, base_model in enumerate(list_of_checkpoints):
        for num_j, model_checkpoint in enumerate(list_of_eigencheckpoints):
            print(f"FT on {base_model} & {model_checkpoint}")

            tv1 = TaskVector(base_checkpoint, base_model).vector
            tv2 = TaskVector(base_checkpoint, model_checkpoint).vector
            differences_1, v1, v2, square_size, weighter = \
                compute_model_distances(tv1, tv2, norm='hist')
            matrix[num_k, num_j] = differences_1  # This is similarity

    df_distance = pd.DataFrame(matrix, index=checkpoints_names, columns=[f"eigenvector {i}-th" for i in range(6)])

    # Create a heatmap using Seaborn
    plt.figure(figsize=(8 * 2, 6 * 2))
    hm = sns.heatmap(df_distance, annot=True, cmap='viridis', fmt='.2f', linewidths=.5)
    hm.set_yticklabels(hm.get_yticklabels(), rotation=45, horizontalalignment='right')
    plt.title('Model Distance Matrix')
    plt.xlabel('EigenVectors')
    plt.ylabel('Models')
    plt.savefig('../TMP_VIZ/eigencosine_distance_scene.png')
    return matrix


def ties_merge_models(base_model_ckpt_pth, models_to_merge_ckpt_pth):
    # TIES Merging example
    K = 60
    merge_func = "dis-sum"
    lamda = 1

    flat_ft = torch.vstack(
        [state_dict_to_vector(torch.load(check), []) for check in models_to_merge_ckpt_pth]
    )

    ptm = torch.load(base_model_ckpt_pth)
    # add back the PTM to the flat merged task vector
    flat_ptm = state_dict_to_vector(ptm, [])
    tv_flat_checks = flat_ft - flat_ptm

    # return merged flat task vector
    merged_tv = ties_merging(
        tv_flat_checks,
        reset_thresh=K,
        merge_func=merge_func,
    )

    merged_check = flat_ptm + lamda * merged_tv

    # convert the flat merged checkpoint to a state dict
    merged_state_dict = vector_to_state_dict(
        merged_check, ptm, remove_keys=[]
    )

    base = '/data2/users/amolina/oda_ocr_output/ties_merge/from_hiertext/'
    os.makedirs(base, exist_ok=True)
    torch.save(merged_state_dict, base + f"ties_merge_k={K}_lambda={lamda}_merge={merge_func}.pth")


if __name__ == '__main__':
    output_tmp_folder = '../TMP_VIZ/'
    os.makedirs(output_tmp_folder, exist_ok=True)

    args = parse_arguments()
    models = MODELS_LUT
    results_per_dataset = {}
    results_per_domain = {}
    print('cosine double check')

    #### COMMON STAGE ####
    # model_distance(models, os.path.join(output_tmp_folder, 'model_distances.png'), 2)
    # compute_cumulative_top_neurons_change(models, 'Zero-Shot')
    # base_model = models.pop('Zero-Shot (hiertext)')
    # task_vectors_cosine_similarity_matrix(base_model, models.values())
    # plot_values_task_vectors( models.pop('Zero-Shot (hiertext)'), models.values())
    zeroshot = models.pop('Zero-Shot (hiertext)')
    compute_svd(zeroshot, models.values())
    # task_vectors_cosine_similarity_matrix(zeroshot, models.values())
    # ties_merge_models(zeroshot, models.values())
    # prepare_dictionary_of_vectors(models, zeroshot)
    # eigencheckpoints = [f'/data2/users/amolina/oda_ocr_output/svd/from_hiertext_sceneonly/{i}_eigenvector.pth'
    #                     for i in range(6)]
    #
    # # task_vectors_cosine_similarity_matrix(zeroshot, models.values(), norm='cosine')
    # task_vectors_cosine_similarity_matrix_eigen(zeroshot, models.values(), eigencheckpoints, models.keys())
