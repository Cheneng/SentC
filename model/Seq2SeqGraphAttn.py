import torch
import torch.nn as nn
import torch.nn.functional as F
from .BasicModule import BasicModule
from .Seq2Seq import *


class LSTMGraphAttn(BasicModule):
    def __init__(self, config):
        super(LSTMGraphAttn, self).__init__()
        self.config = config
        self.encoder = Encoder(config)
        self.decoder = Decoder(config)
        self.graph_attn = GraphAttnNN(config)

        self.linear_attn = nn.Linear(2 * config.de_hidden_size, config.en_input_size)
        self.linear_out = nn.Linear(3 * config.en_input_size, config.de_input_size)

    def forward(self, src, trg, graph_neighbor, hidden=None):
        batch_size = src.shape[0]
        encoder_out, hidden = self.step_encoding(src, hidden)   # Get the hidden state of the encoder
        result = []
        atten_reslut = []
        for index in range(trg.shape[1]):
            trg_step = trg[:,index,:].view(batch_size, 1, -1)
            neighbor = torch.index_select(graph_neighbor, 1, torch.tensor(index).cuda())    # TODO: 维度正确？

            graph_last_input = torch.index_select(encoder_out, 1, torch.tensor(self.config.max_len-1-index).cuda())

            # graph attention
            graph_attn_out, _ = self.graph_attn(neighbor.squeeze(1), hidden, graph_last_input)    # plus for attention
            # graph_attn_out = torch.index_select(graph_attn_out, 1, torch.tensor(-1).cuda())
            graph_attn_out = graph_attn_out[:, -1, :].view(batch_size, 1, -1)
            # 结合前一层得到的hidden state和图网络结果得到attention
            # attn = torch.cat([hidden[1].view(batch_size, 1, -1), graph_attn_out], dim=-1)
            # attn = F.relu(self.linear_attn(attn))
            # 结合当前的输入和attention得到最终的输出

            trg_step = torch.cat([trg_step, graph_attn_out], dim=-1)
            trg_step = F.relu(self.linear_out(trg_step))

            out, hidden = self.step_decoding(trg_step, hidden)
            result.append(out)
        out = torch.cat(result, dim=-1)

        return out

    def step_encoding(self, encoder_input, hidden=None):
        output, hidden = self.encoder(encoder_input, hidden)
        hidden = tuple([i.view(1, -1, self.config.de_hidden_size) for i in hidden])
        return output, hidden

    def step_decoding(self, decoder_input, hidden=None):
        if hidden is None:
            raise ValueError("The hidden should not be None")
        output, hidden = self.decoder(decoder_input, hidden)
        return output, hidden


class GraphAttnNN(nn.Module):
    def __init__(self, config):
        super(GraphAttnNN, self).__init__()
        self.config = config
        self.attn_rnn = nn.GRU(input_size=config.en_input_size,
                               hidden_size=config.de_input_size,
                               batch_first=True)
    def forward(self, graph_neighbor, hidden, last_input=None):
        out, hidden = self.attn_rnn(graph_neighbor, hidden[1])
        if last_input is not None:
            out, hidden = self.attn_rnn(last_input[:,:, -100:], hidden)

        return out, hidden

