import torch
from tqdm import tqdm
import numpy as np

from src.Datafix.datafix import DFCorrect, DFLocate

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

        self.datafix_locate = DFLocate(verbose=True).shift_location(src_tokens, query_tokens)
        self.corruption_mask = torch.tensor(self.datafix_locate.mask_).to(self.encoder.device)
        print('Total corrputed features:', self.datafix_locate.n_corrupted_features_ ,
              'out of', len(self.corruption_mask))
        # self.datafix_correct = DFCorrect(self.datafix_locate.n_corrupted_features_,
        #                                verbose=True).fit_transform(src_tokens, query_tokens)

    def get_tokens_from_dataset(self, dataset, collator, max_tokens = 10000):
        dataloader_tmp = torch.utils.data.DataLoader(dataset, shuffle=False, batch_size = 1,
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

    def compute_by_dropout(self, batch):
        '''
        Takes the mask and drops out all corruputed features, can run 'on-line' and it's the default or forward.
        The sparser the distribution, the harder the task.
        '''
        output_tokens = self.encoder(batch)['features'] * (1-self.corruption_mask).view(1, 1, -1)

        return self.forward_encoded_output(encoder_output=output_tokens)

    def compute_by_applied_shift(self, idx):
        '''
        Using datafix, computes the result with fixed vectors given the corresponding index.
            TODO: Implement DataFix - Correct in order to do this.
        '''
        pass


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
        return self.compute_by_dropout(x)