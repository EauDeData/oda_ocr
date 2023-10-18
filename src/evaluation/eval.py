import torch

from torchmetrics.text import CharErrorRate
from torchmetrics.text import EditDistance
from torchmetrics.text import MatchErrorRate

def clean_special_tokens(string, tokenizer):
    
    for token in tokenizer.special_tokens:
        
        string = string.replace(token, '')
    
    return string
    

def eval_dataset(dataloader, model, dataset_name, tokenizer, wandb_session):
    
    cer = CharErrorRate()
    ed = EditDistance()
    mer = MatchErrorRate()
    
    metrics = {
        f"CER_{dataset_name}": 0,
        f"ED_{dataset_name}": 0,
        f"MER_{dataset_name}": 0    
    }
    total_steps = 0
    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            
            tokens = model(batch)

            argmaxed_output = tokens.argmax(dim = 2).cpu()

            strings = [x for x in tokenizer.decode(argmaxed_output)]
            labels = [x for x in tokenizer.decode(batch['labels'].permute(1, 0))]

            metrics[f"CER_{dataset_name}"] += cer(strings, labels).item()
            metrics[f"ED_{dataset_name}"] += ed(strings, labels).item()
            metrics[f"MER_{dataset_name}"] += mer(strings, labels).item()
            total_steps += 1
        
    final_scores = {key: metrics[key] / total_steps for key in metrics}
    wandb_session.log(
        final_scores
    )
    return final_scores
            
        