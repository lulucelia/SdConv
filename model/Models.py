''' Define the Transformer model '''
import torch
import torch.nn as nn
import numpy as np
from model.Layers import EncoderLayer
import torch.nn.functional as F
import copy


def get_sinusoid_encoding_table(n_position, d_model, padding_idx=None):
    ''' Sinusoid position encoding table
    n_position is the longest video sequence'''

    def cal_angle(position, hid_idx):
        return position / np.power(10000, 2 * (hid_idx // 2) / d_model)

    def get_posi_angle_vec(position):
        return [cal_angle(position, hid_j) for hid_j in range(d_model)]

    sinusoid_table = np.array([get_posi_angle_vec(pos_i) for pos_i in range(n_position)])
    # the even position using sin encoding, and the odd position using cos
    # represent the absolute and the relative position of each frame
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    if padding_idx is not None:
        # zero vector for padding dimension
        sinusoid_table[padding_idx] = 0.

    return torch.FloatTensor(sinusoid_table)


# The dilated conv layers are adapted from: https://github.com/yabufarha/ms-tcn/blob/master/model.py
class DilatedResidualLayer(nn.Module):
    def __init__(self, dilation, in_channels, out_channels):
        super(DilatedResidualLayer, self).__init__()
        self.conv_dilated = nn.Conv1d(in_channels, out_channels, 3, padding=dilation, dilation=dilation)
        self.conv_1x1 = nn.Conv1d(out_channels, out_channels, 1)
        self.dropout = nn.Dropout()

    def forward(self, x):
        out = F.relu(self.conv_dilated(x))
        out = self.conv_1x1(out)
        out = self.dropout(out)
        return (x + out)


class SingleStageModel(nn.Module):
    def __init__(self, n_dlayers, num_f_maps, d_model):
        super(SingleStageModel, self).__init__()
        self.conv_1x1 = nn.Conv1d(d_model, num_f_maps, 1)
        self.layers = nn.ModuleList(
            [copy.deepcopy(DilatedResidualLayer(2 ** i, num_f_maps, num_f_maps)) for i in range(n_dlayers)])

    def forward(self, src_video_seq):
        out = self.conv_1x1(src_video_seq)
        for layer in self.layers:
            out = layer(out)
        return out


# Self-attention block is adpted from https://github.com/jadore801120/attention-is-all-you-need-pytorch
class Encoder(nn.Module):
    ''' A encoder model with self attention mechanism. '''

    def __init__(
            self,
            n_position, n_layers, n_head, d_k, d_v,
            d_model, d_inner, dropout=0.1):

        super().__init__()

        self.position_enc = nn.Embedding.from_pretrained(
            get_sinusoid_encoding_table(n_position, d_model, padding_idx=None),
            freeze=True)

        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)])

    def forward(self, src_video_seq, input_len, return_attns=False):

        enc_slf_attn_list = []

        enc_output = src_video_seq + self.position_enc(input_len)
        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(
                enc_output)
            if return_attns:
                enc_slf_attn_list += [enc_slf_attn]

        if return_attns:
            return enc_output, enc_slf_attn_list
        return enc_output,


class SDConv(nn.Module):
    ''' A sequence to sequence model with attention mechanism. '''

    def __init__(
            self,
            n_position, n_classes, n_dlayers=10, num_f_maps=128, d_model=128, d_inner=512,
            n_layers=6, n_head=8, d_k=16, d_v=16, dropout=0.1):
        super().__init__()

        self.dilation_enc = SingleStageModel(n_dlayers=n_dlayers, num_f_maps=num_f_maps, d_model=d_model)
        self.pool = nn.MaxPool1d(4)
        self.encoder = Encoder(
            n_position=n_position, d_model=num_f_maps, d_inner=d_inner,
            n_layers=n_layers, n_head=n_head, d_k=d_k, d_v=d_v,
            dropout=dropout)
        self.dilation_dec = SingleStageModel(n_dlayers=n_dlayers, num_f_maps=num_f_maps, d_model=num_f_maps)
        self.action_prj = nn.Conv1d(num_f_maps, n_classes, 1)

    # set the return_attns=True, return the self attention value
    def forward(self, src_video_seq, input_len, attn_len):
        # src_video_seq[batch_size, video_length, d_model]
        src_video_seq = src_video_seq.permute(0, 2, 1)
        enc_input = self.dilation_enc(src_video_seq)
        enc_input = self.pool(enc_input).permute(0, 2, 1)
        enc_output, self_attentions = self.encoder(enc_input, attn_len, return_attns=True)
        '''
        enc_output dimension[batch_size(1), video_length, d_model]
        After transpose:[B, d_model, video_length]
        After average pooling[1, d_model, 1]
        '''
        out = enc_output.permute(0, 2, 1)
        out = F.interpolate(out, size=len(input_len[0]))
        out = self.dilation_dec(out)
        out = self.action_prj(out).permute(0, 2, 1)

        return out
        # return out, self_attentions
