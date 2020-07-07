import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

from .Sub_layer import PositionwiseFeedForward, EncoderLayer, DecoderLayer
from .Attention import MultiHeadedAttention
from .Attention2 import dec_MultiHeadedAttention
from .Embedding import Embeddings, PositionalEncoding
from .Enc_Dec import Encoder, Decoder, Generator, EncoderDecoder


def make_model(token_size, N=6, 
               d_model=512, d_ff=2048, h=8, dropout=0.1):
    "Helper: Construct a model from hyperparameters."
    c = copy.deepcopy
    attn = MultiHeadedAttention(h, d_model)
    attn2 = dec_MultiHeadedAttention(h, d_model)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, dropout)
    model = EncoderDecoder(
        Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
        Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
        Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), N),
        nn.Sequential(Embeddings(d_model, token_size), c(position)),
        nn.Sequential(Embeddings(d_model, token_size), c(position)),
        nn.Sequential(Embeddings(d_model, token_size), c(position)),
        Generator(d_model, token_size))
    
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    # model.encoder_embed[0].lut.weight = model.decoder_embed[0].lut.weight
    # model.encoder_embed[0].time_embed.weight = model.decoder_embed[0].time_embed.weight

    return model