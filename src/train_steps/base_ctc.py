import torch
from tqdm import tqdm

def train_ctc(epoch, dataloader, optimizer, model, loss_function, patch_width, wandb_session):
    
    buffer = 0
    counter = 0
    
    model.train()
    for batch in tqdm(dataloader, desc=f"Training classic approach - epoch {epoch}"):
        
        optimizer.zero_grad()
        
        softmaxed_output = model(batch)
        
        ground_truth = batch['labels']
        
        loss = loss_function(softmaxed_output, ground_truth, batch['input_lengths'], batch['output_lengths'])
        
        loss.backward()
        optimizer.step()
        
        counter += 1
        buffer += loss.item()
        
        wandb_session.log({'batch loss': loss.item()})
    wandb_session.log({'train loss': buffer / counter})
        