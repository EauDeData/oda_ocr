import torch
from tqdm import tqdm

def train_cross_entropy(epoch, dataloader, optimizer, model, loss_function, patch_width, wandb_session):
    
    buffer = 0
    counter = 0
    model.train()
    for batch in tqdm(dataloader, desc=f"Training classic approach - epoch {epoch}"):
        
        optimizer.zero_grad()
        
        predicted_seq_logits = model(batch)

        target_seq = batch['labels'].to(model.device).view(-1) # (BS)

        predicted_seq_logits = predicted_seq_logits.permute(1, 0, 2).reshape(predicted_seq_logits.shape[0]*predicted_seq_logits.shape[1], predicted_seq_logits.shape[2])
        
        loss = loss_function(predicted_seq_logits, target_seq)

        if loss != loss:

            images = batch['images_tensor'][0].squeeze().cpu().numpy().transpose(1,2,0)
            import matplotlib.pyplot as plt
            plt.imshow(images)
            plt.title(batch['raw_text_gt'][0])
            plt.savefig('tmp.png')
            batch['pre_resize_images'][0].save('la_temportalidad_del_ser.png')
            exit()


        loss.backward()
        optimizer.step()
        
        counter += 1
        buffer += loss.item()
        
        wandb_session.log({'batch loss': loss.item()})
            
    wandb_session.log({'train loss': buffer / counter})
        