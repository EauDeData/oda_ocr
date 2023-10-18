import torch
from tqdm import tqdm

def train_cross_entropy(epoch, dataloader, optimizer, model, loss_function, patch_width, wandb_session):
    
    buffer = 0
    counter = 0
    
    model.train()
    for batch in tqdm(dataloader, desc=f"Training classic approach - epoch {epoch}"):
        
        optimizer.zero_grad()
        
        predicted_seq_logits = model(batch)

        target_seq = batch['labels'].to(model.device)
        
        loss = loss_function(predicted_seq_logits.view(-1, predicted_seq_logits.shape[-1]), target_seq.view(-1))
        if loss == loss:
        
            loss.backward()
            optimizer.step()
            
            counter += 1
            buffer += loss.item()
            
            wandb_session.log({'batch loss': loss.item()})
            
    wandb_session.log({'train loss': buffer / counter})
        