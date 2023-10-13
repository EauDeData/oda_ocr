import torch
import torch.nn as nn
import math
from torch import Tensor


def linear_constructor(topology: list):

    seq = []
    for n, size in enumerate(topology[:-1]):
        seq.extend([
            nn.ReLU(),
            nn.Linear(size, topology[n + 1])
        ])
    
    return seq

class LinearConstructor(nn.Module):
    def __init__(self, topology, device) -> None:
        super().__init__()
        
        self.layers = [layer.to(device) for layer in linear_constructor(topology)]
    
    def forward(self, x):
        
        for layer in self.layers:
            
            x = layer(x)
        
        return x


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class ViTEncoder(nn.Module):
    def __init__(self, image_height = 128, patch_width = 16, nchannels = 3, token_size = 224, visual_tokenizer_arch = [], nlayers = 6, nheads = 8, vocab_size = None, dropout = 0.1, device = 'cuda'):
        super(ViTEncoder, self).__init__()
        
        self.visual_tokenizer = LinearConstructor(
                [image_height * nchannels * patch_width] + visual_tokenizer_arch + [token_size], device
            )
    
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=token_size, nhead=nheads, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=nlayers) # Out: (BATCH_SIZE, SEQ_SIZE, TOKEN_SIZE)
        
        self.lm_head = nn.Linear(token_size, vocab_size) # Out: (BATCH_SIZE, SEQ_SIZE, TOKEN_SIZE)
        self.lm_softmax = nn.Softmax(dim = 2)

        self.positional_encoding = PositionalEncoding(token_size, dropout=dropout)
        self.device = device
    
    def forward(self, input_dict):
        
        input_visual_tokens = input_dict['input_visual_seq'].to(self.device)
        batch_size, seq_len, channels, width, height = input_visual_tokens.shape
        
        flat_tokens = input_visual_tokens.view(batch_size, seq_len, channels * width * height).permute(1, 0, 2)

        tokenizer_patches = self.visual_tokenizer(flat_tokens)
        positional_encoded_tokens = self.positional_encoding(tokenizer_patches)

        context_tokens = self.transformer_encoder(positional_encoded_tokens)
        
        logits = self.lm_head(context_tokens)
        
        return self.lm_softmax(logits)
        