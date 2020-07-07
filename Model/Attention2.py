import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .Tool_model import clones

def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k, seq_size = query.size(-1), value.size(-2)
    scores = torch.matmul(query, key.transpose(-2, -1)) \
             / math.sqrt(d_k)

    time = torch.exp(-torch.arange(1, seq_size+1).float().unsqueeze(0).unsqueeze(1).unsqueeze(1)).cuda()
            
    if mask is not None:
        scores = scores.masked_fill_(mask == 0, -1e9)
    
    p_attn = F.softmax(scores-time, dim=-1)

    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn

def addictive_attention(query, key, value, U, H, v, mask=None, dropout=None):

    print("query", query.size())
    print("key", key.size())
    print("value", value.size())
    # print("U", U.size())
    # print("H", H.size())
    print("v", v.size())
    exit()

    scores = torch.tanh(U(query)+H(key)).bmm(v.unsqueeze(2))  

    if mask is not None:
        scores = scores.masked_fill_(mask == 0, -1e9)
    
    p_attn = F.softmax(scores, dim = -1)

    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn

class dec_MultiHeadedAttention(nn.Module):

    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(dec_MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

        # self.fc_U = nn.Linear(d_model, d_model, bias=False)
        # self.fc_H = nn.Linear(d_model, d_model, bias=False)
        # self.weight = nn.Parameter(torch.FloatTensor(1, d_model))
        
    def forward(self, query, key, value, mask=None):
        "Implements Figure 2"
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)
        
        # 1) Do all the linear projections in batch from d_model => h x d_k 
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]
        
        # 2) Apply attention on all the projected vectors in batch. 
        x, self.attn = attention(query, key, value, mask=mask, 
                                 dropout=self.dropout)
        # x, self.attn = addictive_attention(query, key, value, 
        #                         U=self.fc_U, H=self.fc_H, v=self.weight, 
        #                         mask=mask, dropout=self.dropout)
        
        # 3) "Concat" using a view and apply a final linear. 
        x = x.transpose(1, 2).contiguous() \
             .view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)


