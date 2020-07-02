import torch
import torch.nn as nn
import torch.nn.functional as F
from .Tool_model import clones, get_mask
from .Sub_layer import LayerNorm


class Encoder(nn.Module):
    "Core encoder is a stack of N layers"
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)
        
    def forward(self, x, mask):
        "Pass the input (and mask) through each layer in turn."
        for layer in self.layers:
            x = layer(x, mask)
        # return self.norm(x)
        return x

class Decoder(nn.Module):
    "Generic N layer decoder with masking."
    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)
        
    def forward(self, x, memory, x_mask, y_mask):
        for layer in self.layers:
            x = layer(x, memory, x_mask, y_mask)
        # return self.norm(x)
        return x

class Generator(nn.Module):
    "Define standard linear + softmax generation step."
    def __init__(self, d_model, token_size):
        super(Generator, self).__init__()
        self.place_Linear = nn.Linear(d_model*2, token_size)

    def forward(self, x):
        logit = self.place_Linear(x)
        return logit.contiguous().view(-1, logit.size(-1))

class EncoderDecoder(nn.Module):
    """
    A standard Encoder-Decoder architecture. Base for this and many 
    other models.
    """
    def __init__(self, encoder, encoder2, decoder, encoder_embed, decoder_embed, encoder2_embed, generator):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.encoder2 = encoder2
        self.encoder_embed = encoder_embed
        self.decoder_embed = decoder_embed
        self.encoder2_embed = encoder2_embed
        self.generator = generator
        
    def forward(self, x, x_time, y, y_time):
        # print(x)
        # print(y)
        # exit()
        x_mask, y_mask, yx_mask = get_mask(x, 'encoder'), get_mask(y, 'decoder'), get_mask(y, 'decoder')

        # print("x",x)
        # print("x_mask",x_mask)
        # exit()

        "Take in and process masked src and target sequences."
        encoder_out = self.encode(x, x_time, x_mask)
        decoder_out = self.decode(encoder_out, x_mask, y, y_time, y_mask)
        # encoder2_output = self.encode2(y, y_time, yx_mask)
        # out = torch.cat([decoder_out, encoder2_output], dim=-1)
        out = decoder_out
        output = self.generator(out)
        return output
    
    def encode(self, x, x_time, x_mask):
        return self.encoder(self.encoder_embed((x, x_time)), x_mask)
    
    def encode2(self, x, x_time, x_mask):
        return self.encoder2(self.encoder_embed((x, x_time)), x_mask)

    def decode(self, memory, x_mask, y, y_time, y_mask):
        return self.decoder(self.encoder_embed((y, y_time)), memory, x_mask, y_mask)
