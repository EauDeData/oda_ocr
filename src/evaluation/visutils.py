import matplotlib.pyplot as plt
import torch


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
