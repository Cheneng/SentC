import torch
import torch.nn as nn
import torch
import torch.nn as nn
import torch.nn.functional as F
from .BasicModule import BasicModule


class LSTM3LayersRes(BasicModule):
    def __init__(self, config):
        super(LSTM3LayersRes, self).__init__()
        self.lstm1 = nn.LSTM(input_size=config.input_size,
                             hidden_size=config.hidden_size,
                             num_layers=config.num_layers,
                             dropout=config.dropout_rate,
                             bidirectional=config.bidirectional,
                             batch_first=config.batch_first)

        self.lstm2 = nn.LSTM(input_size=config.sec_input_size,
                             hidden_size=config.sec_hidden_size,
                             num_layers=config.sec_num_layers,
                             dropout=config.sec_dropout_rate,
                             bidirectional=config.sec_bidirectional,
                             batch_first=config.batch_first)

        self.lstm3 = nn.LSTM(input_size=config.last_input_size,
                             hidden_size=config.last_hidden_size,
                             num_layers=config.last_num_layers,
                             dropout=config.last_dropout_rate,
                             bidirectional=config.bidirectional,
                             batch_first=config.batch_first)


        self.linear_out = nn.Linear(in_features=2*config.last_hidden_size if config.last_bidirectional else config.hidden_size,
                                    out_features=2)

    def forward(self, src, trg, hidden=None):
        _, hidden_encode = self.step_encoding(src, hidden=hidden)
        output, hidden = self.step_decoding(trg, hidden_encode)
        return output, hidden

    def step_encoding(self, src, hidden=None):
        output1, hidden1 = self.lstm1(src, hidden)
        output2, hidden2 = self.lstm2(output1, hidden)
        last_src = torch.cat([output2, src], dim=2)
        output, hidden3 = self.lstm3(last_src, hidden)
        hidden = [hidden1, hidden2, hidden3]
        return output, hidden

    def step_decoding(self, src, hidden=None):
        if hidden is None:
            raise ValueError("The decoding hidden should not be None")
        output1, hidden1 = self.lstm1(src, hidden[0])
        output2, hidden2 = self.lstm2(output1, hidden[1])
        last_src = torch.cat([output2, src], dim=2)
        output3, hidden3 = self.lstm3(last_src, hidden[2])
        output = self.linear_out(output3)
        hidden = [hidden1, hidden2, hidden3]
        return output, hidden

