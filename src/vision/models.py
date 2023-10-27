import torch
import torch.nn as nn
import math
from torch import Tensor
import clip

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
        self.lm_softmax = nn.LogSoftmax(dim = -1)

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
        
        return logits

class ConvVitEncoder(nn.Module):

    def __init__(self, image_height = 128, patch_width = 16, nchannels = 3, token_size = 224, stride = 8, nlayers = 6, nheads = 8, vocab_size = None, dropout = 0.1, device = 'cpu', aggregation = 'max'):
        super(ConvVitEncoder, self).__init__()
        self.image_height = image_height
        self.patch_width = patch_width

        self.conv_1 = nn.Conv2d(nchannels, token_size, kernel_size = (image_height, patch_width), stride = stride)
        
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=token_size, nhead=nheads, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=nlayers) # Out: (BATCH_SIZE, SEQ_SIZE, TOKEN_SIZE)
        self.positional_encoding = PositionalEncoding(token_size, dropout=dropout)
        
        self.lm_head = nn.Linear(token_size, vocab_size) # Out: (BATCH_SIZE, SEQ_SIZE, TOKEN_SIZE)

        self.visual_bos, self.visual_eos = nn.Linear(1, token_size), nn.Linear(1, token_size)

        self.device = device

    def forward(self, input_dict):
        
        input_visual_tensor = input_dict['images_tensor'].to(self.device) # full images (BS, CHANNEL, HEIGHT, WIDTH)

        convolved_tokens = self.conv_1(input_visual_tensor) # tokens (BS, TOKEN_SIZE, 1, SEQ_SIZE)
        unlinearized_tokens = torch.nn.functional.relu(convolved_tokens)

        tokens = unlinearized_tokens.permute(2, 3, 0, 1)[0] # tokens (SEQ_SIZE; BS; TOKEN_SIZE)

        ones = torch.ones(1, tokens.shape[1], 1).to(self.device)
        visual_bos = self.visual_bos(ones)
        visual_eos = self.visual_eos(ones)
        
        tokens_with_bos_and_eos = torch.cat((visual_bos, tokens, visual_eos), dim=0)

        positional_encoded_tokens = self.positional_encoding(tokens_with_bos_and_eos)

        context_tokens = self.transformer_encoder(positional_encoded_tokens)
        
        logits = self.lm_head(context_tokens)
        
        return logits

class FullyConvolutionalEncoder(nn.Module):

    def __init__(self, image_height = 128, patch_width = 16, nchannels = 3, token_size = 224, stride = 8, nlayers = 6, nheads = 8, vocab_size = None, dropout = 0.1, device = 'cpu', aggregation = 'max'):
        super(ConvVitEncoder, self).__init__()


class _ProtoModel(torch.nn.Module):
    def __init__(self, model, device):
        super(_ProtoModel, self).__init__()
        self.model = model
        self.device = device

    def forward(self, x):
        return self.model(x['totally_padded_image'].to(self.device))

class CLIPWrapper(torch.nn.Module):
    def __init__(self, vocab_size, patch_size = 32, device = 'cuda'):
        super(CLIPWrapper, self).__init__()
        model, _ = clip.load(f"ViT-B/{patch_size}", device='cpu')

        self.model = model.visual
        self.input_resolution = self.model.input_resolution
        self.output_dim = self.model.output_dim

        self.conv1 = self.model.conv1
        self.class_embedding = self.model.class_embedding

        self.positional_embedding = self.model.positional_embedding
        self.ln_pre = self.model.ln_pre
        self.transformer = self.model.transformer

        self.lm_head = torch.nn.Linear(768, vocab_size)
        self.device = device

    def forward(self, x):

        x = self.conv1(x['images_tensor'].to(self.device))  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype,
                                                                      device=x.device), x],
                      dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = self.lm_head(x)

        return x

if __name__ == '__main__':

    input_dictionary = {
        'images_tensor': torch.rand(3, 3, 128, 16 * 5),
        'totally_padded_image': torch.rand(3, 3, 224, 224)
    }

    encoder = CLIPWrapper(100, 16, 'cpu')
    print(encoder(input_dictionary).shape)