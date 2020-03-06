import torch
import torch.nn as nn
import torch.nn.functional as F
from BasicModule import BasicModule


class SyntaxLSTM(BasicModule):
    def __init__(self, input_size, hidden_size, tag_size):
        super(SyntaxLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.de_hidden_size = 2 * self.hidden_size
        self.tag_size = tag_size
        self.encoder = nn.LSTM(input_size=self.input_size,
                               hidden_size=self.hidden_size,
                               batch_first=True,
                               bidirectional=True)
        self.lstm = nn.LSTMCell(input_size=self.input_size,
                                hidden_size=self.de_hidden_size)
        self.linear_out = nn.Linear(self.de_hidden_size, 2)
        self.syntax_gate = nn.Linear(self.de_hidden_size + self.tag_size,
                                     self.de_hidden_size)

        self.syntax_embed = nn.Embedding(17, tag_size)
        self.syntax_embed.weight.requires_grad = False


    def forward(self, src, x, tag, hidden=None):
        # if x.size(1) != tag.size():
        #     raise ValueError("The size of x and tag must "
        #                      "be the same but found {0} and {1}".format(x.size, tag.size()))
        b_size = x.size(0)
        seq_len = x.size(1)

        # encode step
        hidden = self.encode_step(src, hidden)
        tag = self.syntax_embed(tag)

        # decode step
        output = []
        for i in range(seq_len):
            x_input, tag_input = x[:, i, :], tag[:, i, :]
            hidden = self.syntax_decoder(x_input, tag_input, hidden)
            output_state = hidden[0]
            output_state = self.linear_out(output_state)
            output.append(output_state.squeeze())

        # output = torch.cat(output, dim=1)
        output = torch.stack(output, dim=1)
        return output

    def syntax_decoder(self, x, tag, hidden):
        """
            run the lstm
        :param x: (batch_size, embedding)
        :param tag: (batch_size, embedding)
        :param hidden: [hidden_state, cell_state]
        :return: (hidden_state, cell_state)
        """
        hidden_state, cell_state = hidden[0][0], hidden[1][0]
        syntax_embed = torch.cat([hidden_state, tag], dim=-1)
        # syntax_attn = F.sigmoid(torch.matmul(self.syntax_gate_param, syntax_embed))
        syntax_attn = F.sigmoid(self.syntax_gate(syntax_embed))
        cell_state = syntax_attn * cell_state

        hidden_state, cell_state = self.lstm(x, (hidden_state, cell_state))
        hidden_state = hidden_state.view(1, -1, 2*self.hidden_size)
        cell_state = cell_state.view(1, -1, 2*self.hidden_size)
        return (hidden_state, cell_state)

    def encode_step(self, src, hidden=None):
        out, hidden = self.encoder(src, hidden)
        # last_hidden = tuple([torch.index_select(state, 0, torch.tensor(1).cuda()) for state in hidden])  # 选择后向的lstm
        last_hidden = (hidden[0].view(1, src.size(0), -1), hidden[1].view(1, src.size(0), -1))
        return last_hidden

    def testing(self, src, trg, tag, embed_flag, hidden=None):
        self.eval()
        b_size = trg.size(0)
        seq_len = trg.size(1)

        # encode step
        hidden = self.encode_step(src, hidden)
        tag = self.syntax_embed(tag)

        # decode step
        output = []
        label = [[2] for j in range(b_size)]
        label = torch.tensor(label).view(b_size, 1).cuda()
        for i in range(seq_len):
            label = embed_flag(label).view(b_size, -1)
            trg_input, tag_input = trg[:, i, :], tag[:, i, :]
            trg_input = torch.cat([trg_input, label], dim=-1)
            hidden = self.syntax_decoder(trg_input, tag_input, hidden)
            output_state = hidden[0]
            output_state = self.linear_out(output_state)

            label = torch.max(output_state, -1)[1]
            output.append(label.view(b_size, -1))
        output = torch.cat(output, dim=-1)
        self.train()
        return output