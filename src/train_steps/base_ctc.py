import torch
from tqdm import tqdm

def train_ctc(epoch, dataloader, optimizer, model, loss_function, patch_width, wandb_session, *args):
    
    buffer = 0
    counter = 0
    
    model.train()
    for batch in tqdm(dataloader, desc=f"Training classic approach - epoch {epoch}"):
        
        optimizer.zero_grad()
        
        softmaxed_output = torch.nn.functional.log_softmax (model(batch)['language_head_output'], dim = -1)
        
        ground_truth = batch['labels']

        # import pdb
        # pdb.set_trace()
        
        
        loss = loss_function(softmaxed_output, ground_truth, tuple(batch['input_lengths']), tuple(batch['output_lengths']))

        loss.backward()
        optimizer.step()
        
        counter += 1
        buffer += loss.item()
        
        wandb_session.log({'batch loss': loss.item()})
    wandb_session.log({'train loss': buffer / counter})


def train_ctc_clip(epoch, dataloader, optimizer, model, loss_function, patch_width, wandb_session, scheduler = None, tokenizer = None, *args):
    buffer = 0
    counter = 0

    model.train()
    for batch in tqdm(dataloader, desc=f"Training classic approach - epoch {epoch}"):

        optimizer.zero_grad()

        softmaxed_output = torch.nn.functional.log_softmax(model(batch)['language_head_output'], dim=-1)

        ground_truth = batch['labels']

        loss = loss_function(softmaxed_output, ground_truth, tuple([softmaxed_output.shape[0] for _ in range(softmaxed_output.shape[1])]),
                             tuple(batch['output_lengths']))

        loss.backward()
        optimizer.step()

        counter += 1
        buffer += loss.item()

        wandb_session.log({'batch loss': loss.item()})
    wandb_session.log({'train loss': buffer / counter, 'lr': scheduler.optimizer.param_groups[0]['lr']})
    scheduler.step(buffer / counter)



