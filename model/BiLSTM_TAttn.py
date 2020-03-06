import torch
import torch.nn as nn
import torch.nn.functional as F
from .BasicModule import BasicModule
from .Seq2Seq import *


class BiLSTMTAttn(BasicModule):
    def __init__(self, config):
        super(BiLSTMTAttn, self).__init__()
        self.config = config
        self.encoder = Encoder(config)
        self.decoder = Decoder(config)

    def forward(self, src, trg, hidden=None):
        encoder_out, hidden = self.step_encoding(src, hidden)
        # hidden仅仅使用逆向的那次lstm，即翻转后的前向
        # hidden = [torch.index_select(state, 0, torch.tensor(0).cuda()) for state in hidden]
        result = []

        for index in range(trg.shape[1]):
            trg_step = torch.index_select(trg, 1, torch.tensor(index).cuda())   # TODO: different mode to use cuda or not
            attn_input = torch.index_select(encoder_out, 1, torch.tensor(self.config.max_len-1-index).cuda())
            trg_step = torch.cat([trg_step, attn_input], dim=2)
            # out, hidden = self.attn_step_decoding(trg_step, encoder_out, hidden)
            out, hidden = self.step_decoding(trg_step, hidden)
            result.append(out)
        out = torch.cat(result, dim=1)
        return out, hidden

    def step_encoding(self, encoder_input, hidden=None):
        output, hidden = self.encoder(encoder_input, hidden)
        # hidden = tuple([i.view(1, -1, self.config.de_hidden_size) for i in hidden])
        # hidden仅仅使用逆向的那次lstm，即翻转后的前向
        hidden = tuple([torch.index_select(state, 0, torch.tensor(0).cuda()) for state in hidden])
        return output, hidden

    def step_decoding(self, decoder_input, hidden=None):
        if hidden is None:
            raise ValueError("The hidden should not be None")
        output, hidden = self.decoder(decoder_input, hidden)
        return output, hidden

    def attn_step_decoding(self, index, one_decode_input, encoder_out, hidden):

        attn_input = torch.index_select(encoder_out, 1, torch.tensor(self.config.max_len-1-index).cuda())
        one_decode_input = torch.cat([one_decode_input, attn_input], dim=2)

        out, hidden = self.step_decoding(one_decode_input, hidden)
        return out, hidden
