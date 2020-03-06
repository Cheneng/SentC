import torch
import torch.nn as nn
import torch.nn.functional as F
from .BasicModule import BasicModule
from .Seq2Seq import *


class BiLSTMSeq2Seq(BasicModule):

    def __init__(self, config):
        super(BiLSTMSeq2Seq, self).__init__()
        self.config = config
        self.encoder = Encoder(config)
        self.decoder = Decoder(config)

    def forward(self, src, trg, hidden=None):
        _, encoder_hidden = self.step_encoding(src, hidden)
        # encoder_hidden = tuple([i.view(self.config.de_num_layers, -1, self.config.de_hidden_size) for i in encoder_hidden])
        out, hidden = self.step_decoding(trg, hidden=encoder_hidden)
        return out, hidden

    def step_encoding(self, encoder_input, hidden=None):
        output, hidden = self.encoder(encoder_input, hidden)
        hidden = tuple([i.view(self.config.de_num_layers, -1, self.config.de_hidden_size) for i in hidden])
        return output, hidden

    def step_decoding(self, decoder_input, hidden=None):
        if hidden is None:
            raise ValueError("The hidden should not be None")
        output, hidden = self.decoder(decoder_input, hidden)
        return output, hidden
