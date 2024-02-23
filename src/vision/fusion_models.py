import torch
import copy
from typing import *
import random

def freeze_module(module):
    for param in module.parameters():
        param.requires_grad = False
    return module

class FusionModelFromEncoderOut(torch.nn.Module):
    # WARNING: Only implemented for vit_atienzaz + TrDecoder
    def __init__(self, base_model: torch.nn.Module, cpoints_list: List[str], emb_size,
                 fusion_width=1, fusion_depth=1, extractor_name='encoder', device='cuda', seq_size = 25,
                 ocr_dropout_chance=0):
        super(FusionModelFromEncoderOut, self).__init__()
        self.ensemble = dict()
        print(f"(Script): Received fusion ensemble for \n\t{' '.join(cpoints_list)}")
        for num, cpoint in enumerate(cpoints_list):

            copied_model = copy.deepcopy(base_model)
            copied_model.load_state_dict(
                torch.load(cpoint)
            )
            if extractor_name is not None:
                module = freeze_module(getattr(copied_model, extractor_name))
            else:
                module = freeze_module(copied_model)

            name = f"{num}_ensemble_module"

            setattr(self, name, module)

            self.ensemble[name] = getattr(self, name)

        encoder_layer = torch.nn.TransformerEncoderLayer(emb_size, fusion_width, batch_first=False)
        self.encoder = torch.nn.TransformerEncoder(encoder_layer, fusion_depth)
        self.seq_encoder_size = seq_size
        #emb_seq_size = 25*len(self.ensemble)+25 if seq_size < 0 else seq_size
        self.embedding = torch.nn.Embedding(25, emb_size)
        self.to(device)
        self.device = device
        self.to_keep_chance = 1-ocr_dropout_chance


    def forward(self, batch):

        if self.training:
            num_ens = max(sum(1 for _ in self.ensemble if random.random() < self.to_keep_chance), 1)
        else:
            num_ens = len(self.ensemble)
        ensembles_to_use = random.sample(list(self.ensemble.values()), num_ens)

        total_features = []
        for model in ensembles_to_use:
            features = model(batch)['features']
            total_features.append(features)

        tensor_of_ints = torch.arange(25).unsqueeze(0).repeat(features.shape[1], 1)\
            .transpose(1,0).to(self.device)
        collected_embs = self.embedding(tensor_of_ints)
        input_to_fuse = torch.cat((collected_embs, *total_features), dim=0)
        output = self.encoder(input_to_fuse)[:self.seq_encoder_size]

        return {
            'features': output,
            'language_head_output': output
        }