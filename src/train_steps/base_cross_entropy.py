import torch
from tqdm import tqdm


def train_cross_entropy(epoch, dataloader, optimizer, model, loss_function, patch_width, wandb_session,
                        padding_token=0, *args, **kwargs):
    buffer = 0
    counter = 0
    model.train()
    for batch in tqdm(dataloader, desc=f"Training classic approach - epoch {epoch}"):
        optimizer.zero_grad()

        predicted_seq_logits = model(batch)['language_head_output']

        target_seq = (batch['padded_labels_to_text'].to(model.device))  # (BS, SEQ_LEN)

        padding = torch.ones(target_seq.shape[0], predicted_seq_logits.shape[0], dtype=target_seq.dtype) * padding_token
        padding[:target_seq.shape[0], :target_seq.shape[1]] = target_seq
        padded_labels = padding.view(-1).to(model.device)

        predicted_seq_logits = predicted_seq_logits.permute(1, 0, 2).reshape(
            predicted_seq_logits.shape[0] * predicted_seq_logits.shape[1], predicted_seq_logits.shape[2])

        if isinstance(loss_function, type(torch.nn.NLLLoss)):
            predicted_seq_logits = torch.nn.functional.log_softmax(predicted_seq_logits, dim = -1)
        loss = loss_function(predicted_seq_logits, padded_labels)
        loss.backward()
        optimizer.step()

        counter += 1
        buffer += loss.item()

        wandb_session.log({'batch loss': loss.item()})

    wandb_session.log({'train loss': buffer / counter})

