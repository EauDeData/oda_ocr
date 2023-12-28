import torch
from tqdm import tqdm
import numpy as np

from datafix import DFLocate
from datafix import DFCorrect

class DataFixTransfer(torch.nn.Module):
    def __init__(self, model, collator, source_dataset, target_dataset, max_tokens = 10000):
        super(DataFixTransfer, self).__init__()
        attributes_to_steal = ('encoder', 'memory', 'projection', 'gelu_fn',
                               'layer', 'decoder', 'lm_head')

        for attr in attributes_to_steal:
            setattr(self, attr, getattr(model, attr)) # Hopefully weights are still there

        print("(DataFix) Succesfully loaded all attributes.")
        query_tokens = self.get_tokens_from_dataset(target_dataset, collator, max_tokens)
        src_tokens = self.get_tokens_from_dataset(source_dataset, collator, max_tokens)
        self.datafix_correct = DFCorrect().fit(src_tokens, query_tokens)

    def get_tokens_from_dataset(self, dataset, collator, max_tokens = 10000):
        dataloader_tmp = torch.utils.data.DataLoader(dataset, shuffle = True, batch_size = 1,
                                                     collate_fn=collator.collate)
        all_tokens = []
        token_count = 0
        with torch.no_grad():
            for batch in tqdm(dataloader_tmp, desc = 'extracting features in datafix...'):
                output_tokens = self.encoder(batch)['features'].cpu()
                numpy_tokens = output_tokens.view(-1, output_tokens.shape[-1]).numpy()
                all_tokens.append(numpy_tokens)
                token_count += numpy_tokens.shape[0]
                if token_count >= max_tokens: break

        return np.vstack(all_tokens)

    def forward_encoded_output(self, encoder_output):
        memory = self.memory(encoder_output)

        projected = self.gelu_fn(self.projection(encoder_output))  # Project encoder output to decoder token size
        tgt_mask = torch.nn.Transformer.generate_square_subsequent_mask(projected.size(0)).to(projected.device)

        # Perform decoding using TransformerDecoder
        decoded = self.gelu_fn(self.decoder(
            tgt=projected,
            memory=memory,
            tgt_mask=tgt_mask
        ))


        # Project the decoder output to vocabulary space
        output = self.lm_head(decoded)

        return {
            'features': decoded,
            'language_head_output': output,
            'hidden_states': None
        }
    def forward(self, x):
        pass