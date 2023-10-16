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
        
        loss = loss_function(softmaxed_output, ground_truth, tuple(batch['input_lengths']), tuple(batch['output_lengths']))
        if loss!=loss:

            print('input_len_declarada')
            print(batch['input_lengths'])

            print('output_len_declarada')
            print(tuple(batch['output_lengths']))

            print('gt tokens:')
            print([x for x in batch['raw_text_gt']])

            print('gt lens:')
            print([len(x) for x in batch['raw_text_gt']] )

            print('input_patches:')
            print([x.size[0] // patch_width for x in batch['original_images']])

            print('input_sizes:')
            print([x.size for x in batch['original_images']])
            batch['original_images'][0].save('tmp_nan.png')

            exit()
        loss.backward()
        optimizer.step()
        
        counter += 1
        buffer += loss.item()
        
        wandb_session.log({'batch loss': loss.item()})
    wandb_session.log({'train loss': buffer / counter})
        