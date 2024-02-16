import wandb
import os
import torch
import networkx as nx
import matplotlib.pyplot as plt

from src.io.args import parse_arguments


from src.io.models_dictionary import MODELS_LUT
from scipy.spatial.distance import cosine
from scipy.stats import wasserstein_distance
def p_norm(v1, v2, p):
    return sum([abs(i - j) ** p for i, j in zip(v1, v2)])**(1/p)
def cosine_ignorer(v1, v2, *args):
    return cosine(v1, v2)
def divergence_ignorer(v1, v2, *args):
    return wasserstein_distance(v1, v2)

NORMS = {'cosine': cosine_ignorer, 'divergence': divergence_ignorer}
def model_distance(models_lut, filepath, norm = 2):

    graph = nx.Graph()
    done_models = []
    for base_model in models_lut:
        for model_checkpoint in [model for model in models_lut if model!=base_model]:
            if (base_model, model_checkpoint) in done_models: continue

            left_model_dict, right_model_dict = torch.load(models_lut[model_checkpoint]),\
                torch.load(models_lut[base_model])
            v1 = []
            v2 = []
            print(base_model, model_checkpoint)
            for left_key, right_key in zip(left_model_dict, right_model_dict):
                assert left_key==right_key, ('Linearizing models is not implemented yet,'
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
            graph.add_edge(base_model, model_checkpoint, weight=weighter(v1, v2, norm))
            done_models.append((model_checkpoint, base_model))

    nx.draw(graph, nx.spring_layout(graph), with_labels=True, node_size=700, node_color="skyblue", font_size=10)
    nx.draw_networkx_edge_labels(graph, nx.spring_layout(graph), edge_labels=\
        {(n1, n2): f"{d['weight']:.2f}" for n1, n2, d in graph.edges(data=True)})

    # Save the graph to a file
    plt.savefig(filepath, format="PNG")
    print({(n1, n2): f"{d['weight']:.2f}" for n1, n2, d in graph.edges(data=True)})
    nx.write_gexf(graph, filepath.replace('.png', '.gexf'))

if __name__ == '__main__':
    output_tmp_folder = '../TMP_VIZ/'
    os.makedirs(output_tmp_folder, exist_ok=True)

    args = parse_arguments()
    models = MODELS_LUT
    results_per_dataset = {}
    results_per_domain = {}

    #### COMMON STAGE ####
    model_distance(models, os.path.join(output_tmp_folder, 'model_distances.png'), 2)