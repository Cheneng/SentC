import torch
import torch.nn as nn
import torch
import torch.nn as nn
import torch.nn.functional as F
from .BasicModule import BasicModule


class LSTM3Layers(BasicModule):
    def __init__(self, config):
        super(LSTM3Layers, self).__init__()
        self.lstm = nn.LSTM(input_size=config.input_size,
                            hidden_size=config.hidden_size,
                            num_layers=config.num_layers,
                            dropout=config.dropout_rate,
                            bidirectional=config.bidirectional,
                            batch_first=config.batch_first)
        self.linear_out = nn.Linear(in_features=2*config.hidden_size if config.bidirectional else config.hidden_size,
                                    out_features=2)

    def forward(self, src, trg, hidden=None):
        _, hidden_encode = self.step_encoding(src, hidden=hidden)
        output, hidden = self.step_decoding(trg, hidden_encode)
        return output, hidden

    def step_encoding(self, src, hidden=None):
        output, hidden = self.lstm(src, hidden)
        return output, hidden

    def step_decoding(self, src, hidden=None):
        output, hidden = self.lstm(src, hidden)
        output = self.linear_out(output)
        return output, hidden

