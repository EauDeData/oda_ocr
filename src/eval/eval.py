import torch

from torchmetrics.text import CharErrorRate
from torchmetrics.text import EditDistance
from torchmetrics.text import MatchErrorRate

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
    with torch.no_grad():
        for batch in dataloader:
            
            tokens = model(batch)
            argmaxed_output = tokens.argmax(dim = 2).cpu()
            strings = tokenizer.decode(argmaxed_output)
            
            metrics[f"CER_{dataset_name}"] += cer(strings, batch['raw_text_gt']).item()
            metrics[f"ED_{dataset_name}"] += ed(strings, batch['raw_text_gt']).item()
            metrics[f"MER_{dataset_name}"] += mer(strings, batch['raw_text_gt']).item()
            total_steps += 1
        
    final_scores = {key: metrics[key] / total_steps for key in metrics}
    wandb_session.log(
        final_scores
    )
    return final_scores
            
        