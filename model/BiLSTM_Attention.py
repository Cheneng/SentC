import torch
import torch.nn as nn
import torch.nn.functional as F
from .BasicModule import BasicModule
from .Seq2Seq import *


class BiLSTMAttn(BasicModule):
    def __init__(self, config):
        super(BiLSTMAttn, self).__init__()
        self.config = config
        self.encoder = Encoder(config)
        self.decoder = Decoder(config)
        self.atten = nn.Linear(config.de_input_size + config.de_hidden_size,
                               config.max_len)
        self.attn_combine = nn.Linear(config.de_input_size + config.de_hidden_size,
                                      config.de_input_size)

    def forward(self, src, trg, hidden=None):
        encoder_out, hidden = self.step_encoding(src, hidden)
        # hidden = tuple([i.view(1, -1, self.config.de_hidden_size) for i in encoder_hidden])
        result = []
        for index in range(trg.shape[1]):
            trg_step = torch.index_select(trg, 1, torch.tensor(index).cuda())   # TODO: different mode to use cuda or not
            out, hidden = self.attn_step_decoding(trg_step, encoder_out, hidden)
            result.append(out)
        out = torch.cat(result, dim=1)
        return out, hidden

    def step_encoding(self, encoder_input, hidden=None):
        output, hidden = self.encoder(encoder_input, hidden)
        hidden = tuple([i.view(1, -1, self.config.de_hidden_size) for i in hidden])
        return output, hidden

    def step_decoding(self, decoder_input, hidden=None):
        if hidden is None:
            raise ValueError("The hidden should not be None")

        output, hidden = self.decoder(decoder_input, hidden)
        return output, hidden

    def attn_step_decoding(self, one_decode_input, encoder_out, hidden):
        # cell_state = hidden[0]
        cell_state = hidden[1]  # the dim=1 is the cell-state
        atten_input = torch.cat([cell_state.view(one_decode_input.shape[0], 1, -1),
                                 one_decode_input], dim=2)
        atten_weight = self.atten(atten_input)
        atten_weight = F.softmax(atten_weight, dim=2)
        attention_get = torch.matmul(atten_weight, encoder_out)
        combine = torch.cat([one_decode_input, attention_get], dim=2)
        src = self.attn_combine(combine)
        out, hidden = self.step_decoding(src, hidden)
        return out, hidden
