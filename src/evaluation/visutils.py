import matplotlib.pyplot as plt
import torch
import numpy as np


def visualize_whats_going_on(input_batch, model, tokenizer, output_path):
    model.eval()
    with torch.no_grad():
        tokens = model(input_batch)['language_head_output'].cpu().argmax(-1).permute(1, 0)
        predicted = [x for x in tokenizer.decode(tokens)]  # Should permute?
        print(predicted)
        import pdb
        pdb.set_trace()
    model.train()


def loop_for_visualization(dataloader, model, tokenizer, output_path):
    for batch in dataloader:
        visualize_whats_going_on(batch, model, tokenizer, output_path)
def plot_radial(labels, numbers_list, names = None):
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    for name, numbers in zip(range(len(numbers_list)) if names is None else names, numbers_list):
        # plt.plot(angles, numbers, alpha=0)
        plt.fill(angles, numbers, alpha=0.3, label=name)

    ax.set_xticks(angles)
    ax.set_xticklabels(labels)
    if not names is None:
        plt.legend()
    plt.show()




