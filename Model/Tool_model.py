import copy
import torch
import torch.nn as nn

def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

def get_mask(x, mode='encoder'):
    len_s = x.size(-1)
    mask_pad = (x != 0).unsqueeze(-2)
    if mode=='encoder':
        mask = mask_pad
    elif mode=='decoder':
        mask_next = (1 - torch.triu(torch.ones((1, len_s, len_s), device=x.device), diagonal=1)).bool()
        mask = mask_pad & mask_next
    return mask