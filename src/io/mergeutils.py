import copy
# using pytorch:
# This are reptile utils for operating with the models
from measure_distances_of_models import from_vector_to_state_dict
from tqdm import tqdm
import torch
from src.task_vectors_original import TaskVector

def weighted_gradient_state_dict(weight, state_dict_0, finetuned_state_dict):
    """
    Computes the weighted difference between two state dictionaries.
    weight * (state_dict_0 - finetuned_state_dict)

    Args:
        weight (float): The weight to scale the difference.
        state_dict_0 (dict): The initial state dictionary.
        finetuned_state_dict (dict): The finetuned state dictionary.

    Returns:
        dict: A new state dictionary containing the weighted differences.
    """
    diff_state_dict = {}
    for key in state_dict_0:
        diff_state_dict[key] = weight * (state_dict_0[key] - finetuned_state_dict[key])
    return diff_state_dict

def sum_state_dicts(list_of_state_dcts: list):
    """
    Sums a list of state dictionaries element-wise.

    Args:
        list_of_state_dcts (list): A list of state dictionaries to sum.

    Returns:
        dict: A new state dictionary that is the sum of all input dictionaries.
    """
    summed_state_dict = copy.deepcopy(list_of_state_dcts[0])
    for key in summed_state_dict:
        for state_dict in list_of_state_dcts[1:]:
            summed_state_dict[key] += state_dict[key]
    return summed_state_dict

def copy_model_and_optimizer(optimizer, model):
    """
    Creates a copy of the model and its optimizer.

    Args:
        optimizer (torch.optim.Optimizer): The optimizer to copy.
        model (torch.nn.Module): The model to copy.

    Returns:
        tuple: A tuple containing the new model instance and the new optimizer instance.
    """
    model_copy = copy.deepcopy(model) # Does it even work this way?
    optimizer_class = optimizer.__class__
    optimizer_state_dict = optimizer.state_dict()

    new_optimizer = optimizer_class(model_copy.parameters(), lr=optimizer.param_groups[0]['lr'])
    new_optimizer.load_state_dict(optimizer_state_dict)

    return new_optimizer, model_copy

def reptile_model_step(previous_model, epsilon_step_size, list_of_ft_models: list):
    """
    Performs a Reptile meta-learning step.

    Args:
        previous_model (str): The original model before fine-tuning.
        epsilon_step_size (float): The epsilon step size for the Reptile algorithm.
        list_of_ft_models (list): A list of models that were fine-tuned on different tasks.

    Returns:
        torch.nn.Module: The updated model after applying the Reptile step.
    """

    models = []

    for num_k, base_model in tqdm(enumerate(list_of_ft_models), total=len(list_of_ft_models)):

        v1 = []
        tv1 = TaskVector(previous_model, base_model).vector
        for left_key in tv1:
            v1.extend(tv1[left_key]
                      .reshape(torch.numel(tv1[left_key])).cpu()
                      .numpy().tolist()
                      )
        t = torch.tensor(v1, device='cuda')
        models.append(t)
    models = torch.stack(models)
    mean_model = models.mean(0) * epsilon_step_size # Normally it is just 1xAVG(models)
    mean_model_state_dict = TaskVector(vector=from_vector_to_state_dict(tv1, mean_model)) \
        .apply_to(previous_model)


    return mean_model_state_dict

