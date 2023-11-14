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
    def __init__(self, image_height=128, patch_width=16, nchannels=3, token_size=224, visual_tokenizer_arch=[],
                 nlayers=6, nheads=8, vocab_size=None, dropout=0.1, device='cuda'):
        super(ViTEncoder, self).__init__()

        self.visual_tokenizer = LinearConstructor(
            [image_height * nchannels * patch_width] + visual_tokenizer_arch + [token_size], device
        )

        self.encoder_layer = nn.TransformerEncoderLayer(d_model=token_size, nhead=nheads, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer,
                                                         num_layers=nlayers)  # Out: (BATCH_SIZE, SEQ_SIZE, TOKEN_SIZE)

        self.lm_head = nn.Linear(token_size, vocab_size)  # Out: (BATCH_SIZE, SEQ_SIZE, TOKEN_SIZE)
        self.lm_softmax = nn.LogSoftmax(dim=-1)

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

        return {'features': context_tokens, 'language_head_output': logits}


class ConvVitEncoder(nn.Module):

    def __init__(self, image_height=128, patch_width=16, nchannels=3, token_size=224, stride=8, nlayers=6, nheads=8,
                 vocab_size=None, dropout=0.1, device='cpu', aggregation='max'):
        super(ConvVitEncoder, self).__init__()
        self.image_height = image_height
        self.patch_width = patch_width

        self.conv_1 = nn.Conv2d(nchannels, token_size, kernel_size=(image_height, patch_width), stride=stride)

        self.encoder_layer = nn.TransformerEncoderLayer(d_model=token_size, nhead=nheads, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer,
                                                         num_layers=nlayers)  # Out: (BATCH_SIZE, SEQ_SIZE, TOKEN_SIZE)
        self.positional_encoding = PositionalEncoding(token_size, dropout=dropout)

        self.lm_head = nn.Linear(token_size, vocab_size)  # Out: (BATCH_SIZE, SEQ_SIZE, TOKEN_SIZE)

        self.visual_bos, self.visual_eos = nn.Linear(1, token_size), nn.Linear(1, token_size)

        self.device = device

    def forward(self, input_dict):
        input_visual_tensor = input_dict['images_tensor'].to(self.device)  # full images (BS, CHANNEL, HEIGHT, WIDTH)

        convolved_tokens = self.conv_1(input_visual_tensor)  # tokens (BS, TOKEN_SIZE, 1, SEQ_SIZE)
        unlinearized_tokens = torch.nn.functional.relu(convolved_tokens)

        tokens = unlinearized_tokens.permute(2, 3, 0, 1)[0]  # tokens (SEQ_SIZE; BS; TOKEN_SIZE)

        ones = torch.ones(1, tokens.shape[1], 1).to(self.device)
        visual_bos = self.visual_bos(ones)
        visual_eos = self.visual_eos(ones)

        tokens_with_bos_and_eos = torch.cat((visual_bos, tokens, visual_eos), dim=0)

        positional_encoded_tokens = self.positional_encoding(tokens_with_bos_and_eos)

        context_tokens = self.transformer_encoder(positional_encoded_tokens)

        logits = self.lm_head(context_tokens)

        return {'features': context_tokens, 'language_head_output': logits}


class FullyConvolutionalEncoder(nn.Module):

    def __init__(self, image_height=128, patch_width=16, nchannels=3, token_size=224, stride=8, nlayers=6, nheads=8,
                 vocab_size=None, dropout=0.1, device='cpu', aggregation='max'):
        super(ConvVitEncoder, self).__init__()


class _ProtoModel(torch.nn.Module):
    def __init__(self, model, device, target='totally_padded_image'):
        super(_ProtoModel, self).__init__()
        self.model = model
        self.device = device
        self.target = target

    def forward(self, x):
        output = self.model(x[self.target].to(self.device))
        return {
            'features': output,
            'language_head_output': output
        }


class CLIPWrapper(torch.nn.Module):
    def __init__(self, vocab_size, patch_size=32, device='cuda'):
        super(CLIPWrapper, self).__init__()
        model, _ = clip.load(f"ViT-B/{patch_size}", device='cpu')

        self.model = model.visual
        self.input_resolution = self.model.input_resolution
        self.output_dim = self.model.output_dim

        self.conv1 = self.model.conv1
        self.class_embedding = self.model.class_embedding

        self.positional_embedding = self.model.positional_embedding  # torch.nn.Parameter(torch.rand((1, 512, 768)))
        self.ln_pre = self.model.ln_pre
        self.transformer = self.model.transformer

        self.lm_head = torch.nn.Linear(768, vocab_size)
        self.device = device
        self.gelu_fn = torch.nn.GELU()

    def forward(self, x):
        x = self.conv1(x['totally_padded_image'].to(self.device))  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype,
                                                                      device=x.device), x],
                      dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)  # Maybe the original positional embedding is too small

        x = self.ln_pre(x)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = self.gelu_fn(x)
        language_head_output = self.lm_head(x)

        return {'features': x, 'language_head_output': language_head_output}


class GenericModule(torch.nn.Module):
    def __init__(self, module):
        super(GenericModule, self).__init__()
        self.vitstr = module

    def forward(self, x):
        return self.vitstr(x)


class ViTAtienzaWrapper(torch.nn.Module):
    def __init__(self, model):
        super(ViTAtienzaWrapper, self).__init__()
        self.module = GenericModule(model)

    def forward(self, x):
        x = x.mean(1).unsqueeze(1)  # Grayscale Image (not true, but will work)
        return self.module(x).permute(1, 0, 2)


class RNNDecoder(nn.Module):
    def __init__(self, encoder, encoder_input_size, decoder_token_size, decoder_depth, vocab_size, kind='lstm'):
        super(RNNDecoder, self).__init__()

        self.encoder = encoder
        self.projection = torch.nn.Linear(encoder_input_size, decoder_token_size)
        self.gelu_fn = torch.nn.GELU()

        if kind == 'lstm':
            self.recurrent_module = torch.nn.LSTM(decoder_token_size, vocab_size, num_layers=decoder_depth)

        elif kind == 'gru':
            self.recurrent_module = torch.nn.GRU(decoder_token_size, vocab_size, num_layers=decoder_depth)

        elif kind == 'rnn':
            self.recurrent_module = torch.nn.RNN(decoder_token_size, vocab_size, num_layers=decoder_depth)

        else:
            raise NotImplementedError(f"{kind} is not an implemented recurrent type.")

    def forward(self, x):
        features = self.encoder(x)['features']

        projected_features = self.gelu_fn(self.projection(features))
        output_sequence_logits, (hn, cn) = self.recurrent_module(projected_features)

        return {
            'features': projected_features,
            'language_head_output': output_sequence_logits,
            'hidden_states': (hn, cn)
        }


class TransformerDecoder(nn.Module):
    def __init__(self, encoder, encoder_input_size, decoder_token_size, decoder_depth, vocab_size, decoder_width):
        super(TransformerDecoder, self).__init__()

        self.encoder = encoder
        self.projection = torch.nn.Linear(encoder_input_size, decoder_token_size)
        self.gelu_fn = torch.nn.GELU()

        self.layer = nn.TransformerDecoderLayer(d_model=decoder_token_size, nhead=decoder_width)
        self.decoder = nn.TransformerDecoder(self.layer, num_layers=decoder_depth)

        self.lm_head = torch.nn.Linear(decoder_token_size, vocab_size)

    def forward(self, X):
        import pdb
        pdb.set_trace()
        encoder_output = self.encoder(X['features'])  # Pass the batch X through the encoder
        projected = self.gelu_fn(self.projection(encoder_output))  # Project encoder output to decoder token size

        tgt_mask = nn.Transformer.generate_square_subsequent_mask(projected.size(0)).to(X.device)

        # Perform decoding using TransformerDecoder
        decoded = self.decoder(
            tgt=projected,
            memory=encoder_output,
            tgt_mask=tgt_mask
        )

        # Project the decoder output to vocabulary space
        output = self.lm_head(decoded)

        return {
            'features': decoded,
            'language_head_output': output,
            'hidden_states': None
        }


if __name__ == '__main__':
    input_dictionary = {
        'images_tensor': torch.rand(3, 3, 128, 16 * 5),
        'totally_padded_image': torch.rand(3, 3, 224, 224)
    }

    encoder = CLIPWrapper(100, 16, 'cpu')
    print(encoder(input_dictionary).shape)
