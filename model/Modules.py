import torch
import torch.nn as nn
import numpy as np

__author__ = "Yu-Hsiang Huang"


class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v, mask=None):
        """
        q: Query, shape[Batch_size, L_q, d_q]
        k: Key, shape[B, L_k, d_k]
        v: Value, shape[B, L_v, d_v]
        temperature: a scale facter
        bmm: a batch matrix-matrix product of matrices stored in batch1 and batch2
        """

        attn = torch.bmm(q, k.transpose(1, 2))
        attn = attn / self.temperature

        if mask is not None:
            attn = attn.masked_fill(mask, -np.inf)

        attn = self.softmax(attn)
        attn = self.dropout(attn)
        output = torch.bmm(attn, v)

        return output, attn
