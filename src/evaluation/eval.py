import torch
import numpy as np

from torchmetrics.text import CharErrorRate
from torchmetrics.text import EditDistance
from torchmetrics.text import MatchErrorRate

from src.decoders.decoders import GreedyTextDecoder
from src.io.formatting_io_ops import bcolors

def clean_special_tokens(string, tokenizer):
    for token in tokenizer.special_tokens:
        string = string.replace(token, '')

    return string


def cer(x, y, reduce_fn=sum, cer_fn=CharErrorRate()):
    return reduce_fn([cer_fn([x_n], [y_n]).item() for x_n, y_n in zip(x, y) if len(y_n)])

def mer(x, y, reduce_fn=sum, mer_fn=MatchErrorRate()):
    return reduce_fn([mer_fn([x_n], [y_n]).item() for x_n, y_n in zip(x, y) if len(y_n)])

def ed(x, y, reduce_fn=sum, ed_fn=EditDistance(), normalize=True):
    return reduce_fn([ed_fn([x_n], [y_n]).item() / (len(y_n) if normalize else 1) for x_n, y_n in zip(x, y)
                      if len(y_n)])

def eval_dataset(dataloader, model, dataset_name, tokenizer, wandb_session):
    decoder = GreedyTextDecoder(False)

    metrics = {
        f"CER_{dataset_name}": 0,
        f"ED_{dataset_name}": 0,
        f"MER_{dataset_name}": 0,
        f"accuracy_{dataset_name}": 0
    }
    total_steps = 0
    model.eval()
    model.cuda()
    model.device='cuda'
    with torch.no_grad():
        for batch in dataloader:
            tokens = model(batch)['language_head_output'].cpu().detach().numpy()
            decoded_tokens = decoder({'ctc_output': tokens}, tokenizer.ctc_blank, None)

            strings = [clean_special_tokens(x.split(tokenizer.eos)[0], tokenizer) for x in
                       tokenizer.decode_from_numpy_list([x['text'] for x in decoded_tokens])]

            labels = [clean_special_tokens(x.split(tokenizer.eos)[0], tokenizer) for x in tokenizer.decode(batch['labels'].permute(1, 0))]

            metrics[f"CER_{dataset_name}"] += cer(strings, labels)
            metrics[f"ED_{dataset_name}"] += ed(strings, labels)
            metrics[f"MER_{dataset_name}"] += mer(strings, labels)
            metrics[f"accuracy_{dataset_name}"] += sum(int(x == y) for x, y in zip(strings, labels) if len(y))

            total_steps += 1
        for x, y in zip(strings, labels):
            print(f"{bcolors.OKGREEN if x==y else bcolors.FAIL}Predicted:{x}, GT: {y}{bcolors.ENDC}")

    final_scores = {key: metrics[key] / len(dataloader.dataset) for key in metrics}
    wandb_session.log(
        final_scores
    )
    return final_scores

def eval_dataset_democracy(dataloader, model, dataset_name, tokenizer, wandb_session):
    pass

def intercept_forward(model, batch):

    model_encoder = model.encoder.model.module.vitstr
    print(model_encoder)
    exit()

def eval_dataset_for_print_mask(dataloader, model, dataset_name, tokenizer, wandb_session):
    decoder = GreedyTextDecoder(False)

    model.eval()
    acc = 0
    total = 0
    with torch.no_grad():
        for batch in dataloader:
            tokens = intercept_forward(model, batch)['language_head_output'].cpu().detach().numpy()

            decoded_tokens = decoder({'ctc_output': tokens}, tokenizer.ctc_blank, None)

            strings = [clean_special_tokens(x.split(tokenizer.eos)[0], tokenizer) for x in
                       tokenizer.decode_from_numpy_list([x['text'] for x in decoded_tokens])]

            labels = [clean_special_tokens(x.split(tokenizer.eos)[0], tokenizer) for x in
                      tokenizer.decode(batch['labels'].permute(1, 0))]
        for x, y in zip(strings, labels):
            print(f"{bcolors.OKGREEN if x == y else bcolors.FAIL}Predicted:{x}, GT: {y}{bcolors.ENDC}")

        acc += sum(int(x == y) for x, y in zip(strings, labels) if len(y))
        total += 1

    return acc / total

