import matplotlib.pyplot as plt
import torch
import numpy as np
import pandas as pd
import seaborn as sns


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


def plot_radial(labels, numbers_list, names=None, ylim=None, inv_data=True, output='tmp.png'):
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    for name, numbers in zip(range(len(numbers_list)) if names is None else names, numbers_list):
        if inv_data: numbers = np.clip(1 - np.array(numbers), 0, None)

        plt.plot(angles, numbers, label=None, linewidth=.7, linestyle='--')
        plt.fill(angles, numbers, alpha=0.3, label=name)
    if ylim is not None:
        ax.set_ylim(ylim)
    ax.set_xticks(angles)
    ax.set_xticklabels(labels)
    if not names is None:
        plt.legend()
        plt.rc('legend', fontsize='x-small')  # using a named size
    plt.savefig(output, transparent=True, dpi=300)
    plt.clf()


def prepare_dataframe(labels, numbers, names):
    rows = []
    for numbers_of_model, model_name in zip(numbers, names):
        for result, label in zip(numbers_of_model, labels):
            rows.append([result, label, model_name])

    return pd.DataFrame(rows, columns=['result', 'dataset', 'approach'])


def plot_bars(labels, numbers_list, names=None, ylim=None, output='tmp_bars.png'):
    numbers_list.append([sum([num[idx] for num in numbers_list]) / (len(numbers_list[0]))
                         for idx in range(len(numbers_list[0]))])
    labels.append('model_average')

    df = prepare_dataframe([x.split('_')[1] for x in labels], numbers_list, names)
    ax = sns.catplot(data=df, kind='bar', x='dataset', y='result', hue='approach', errorbar='sd', alpha=.6)
    ax.set_axis_labels("", "Performance")
    ax.despine(left=True)
    # Customize the plot
    # ax.legend(fontsize='small')
    ax.legend.set_title("")
    # ax.set_title('Barplot of Numbers Grouped by Model and Dataset')
    #plt.xticks(rotation=-15)  # Rotate x-axis labels by 45 degrees

    # Show the plot
    plt.grid(True)
    plt.savefig(output, transparent=True, dpi=300)
    plt.clf()
