import torch
import torch.nn as nn
from .BasicModule import BasicModule

class SemanticDiscriminator(BasicModule):
    """
    Using the hidden to judge the semantic similiarity.
    """
    def __init__(self, config):
        super(SemanticDiscriminator, self).__init__()
        self.rnn_origin = nn.GRU(input_size=config.input_dim,
                                 hidden_size=config.hidden_dim,
                                 num_layers=config.num_layers,
                                 bidirectional=config.bidirectional)
        self.rnn_short = nn.GRU(input_size=config.input_dim,
                                hidden_size=config.hidden_dim,
                                num_layers=config.num_layers,
                                bidirectional=config.bidirectional)
        self.dropout = nn.Dropout(p=config.dropout_rate)
        self.linear_1 = nn.Linear(in_features=config.hidden_dim*4,
                                  out_features=config.hidden_dim*4)
        self.linear_out = nn.Linear(in_features=config.hidden_dim*4,
                                    out_features=config.out_class)

    def forward(self, sent1, sent2):
        """

        :param sent1: The origin sentence
        :param sent2: The shorten sentence
        :return:
        """
        out1, hidden1 = self.rnn_origin(sent1)
        out2, hidden2 = self.rnn_short(sent2)

        out1 = out1[:, -1, :]
        out2 = out2[:, -1, :]
        # hidden2 = self.dropout(hidden2)
        out = torch.cat([out1, out2], dim=-1)
        out = self.dropout(out)
        out = self.linear_1(out)
        out = self.dropout(out)
        out = self.linear_out(out)
        return out

