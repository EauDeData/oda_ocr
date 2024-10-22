import torch
from tqdm import tqdm
from copy import deepcopy
import os
from uuid import uuid4
# from src.eval_single_task import ft_encoder
from src.io.mergeutils import (reptile_model_step)


def train_ctc(epoch, dataloader, optimizer, model, loss_function, patch_width, wandb_session, *args, **kwargs):
    
    buffer = 0
    counter = 0
    
    model.train()
    for batch in tqdm(dataloader, desc=f"Training classic approach - epoch {epoch}"):
        
        optimizer.zero_grad()
        
        softmaxed_output = torch.nn.functional.log_softmax (model(batch)['language_head_output'], dim = -1)
        
        ground_truth = batch['labels']

        loss = loss_function(softmaxed_output, ground_truth, tuple(batch['input_lengths']), tuple(batch['output_lengths']))


        loss.backward()
        optimizer.step()
        
        counter += 1
        buffer += loss.item()
        
        wandb_session.log({'batch loss': loss.item()})
    wandb_session.log({'train loss': buffer / counter})


def train_ctc_clip(epoch, dataloader, optimizer, model, loss_function, patch_width, wandb_session, scheduler = None, tokenizer = None, max_steps = None, batch_size=None, data_part='',  return_stuff=False, *args, **kwargs):
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

        wandb_session.log({'batch loss' + f'_{data_part}' if len(data_part) else '': loss.item()})
        if (max_steps is not None) and ((counter * batch_size) > max_steps): break

    wandb_session.log({'train loss' + f'_{data_part}' if len(data_part) else '': buffer / counter})
    if return_stuff:
        return model.state_dict()

def train_reptile(epoch, dataloaders: list, optimizer, model, loss_function, patch_width, wandb_session, reptile_args=None, scheduler = None, tokenizer = None, max_steps = None, batch_size=None, *args, **kwargs):

    os.makedirs(reptile_args.reptile_tmp_folder,  exist_ok=True)
    original_state_dict_fname = os.path.join(reptile_args.reptile_tmp_folder, str(uuid4()) + '_original_model')
    # therefore its state dict is on cpu
    # in reptile the input model is a state dict
    torch.save(model, original_state_dict_fname)
    del model

    updated_models = []
    for n, data_domain in enumerate(dataloaders):

        print(f'Training datum {n} / {len(dataloaders)}: {data_domain.dataset.name}')
        # copied_optimizer, copied_model = copy_model_and_optimizer(optimizer, model)
        copied_model = reptile_args.prepare_model(reptile_args.vocabulary_size, reptile_args)
        copied_model.load_state_dict(torch.load(original_state_dict_fname))
        copied_model.train()

        copied_optimizer = torch.optim.Adam(copied_model.parameters(), lr=reptile_args.learning_rate)

        for epis in range(reptile_args.reptile_num_episodes):
            copied_model_state = train_ctc_clip(epis, data_domain, copied_optimizer, copied_model, loss_function, patch_width, wandb_session, scheduler,
                           tokenizer, max_steps, batch_size, data_part=data_domain.dataset.name, return_stuff=True)

            copied_model.load_state_dict(copied_model_state)
            copied_optimizer = torch.optim.Adam(copied_model.parameters(), lr=reptile_args.learning_rate)

        fname = os.path.join(reptile_args.reptile_tmp_folder, str(uuid4())+f'_{data_domain.dataset.name}')
        torch.save(copied_model.state_dict(), fname)
        updated_models.append(fname)
        del copied_model


    new_model = reptile_model_step(original_state_dict_fname, reptile_args.outer_lr, updated_models)

    return new_model
