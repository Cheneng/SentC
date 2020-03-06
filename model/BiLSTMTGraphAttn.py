import torch
import torch.nn as nn
import torch.nn.functional as F
from .BasicModule import BasicModule
from .Seq2Seq import *


class BiLSTMTGAttn(BasicModule):
    def __init__(self, config):
        super(BiLSTMTGAttn, self).__init__()
        self.config = config
        self.encoder = Encoder(config)
        self.decoder = Decoder(config)
        self.matrix_list = nn.ParameterList([nn.Parameter(torch.randn(config.en_hidden_size * 2, config.en_input_size)) for _ in range(config.degree_num + 1)])

    def forward(self, src, trg, degree, hidden=None):
        batch_size = src.size(0)
        encoder_out, hidden = self.step_encoding(src, hidden)
        # hidden仅仅使用逆向的那次lstm，即翻转后的前向
        result = []
        # temp_matrix = torch.empty(batch_size, self.config.en_hidden_size * 2,
        #                           self.config.en_hidden_size)  # save the temp matrix for this situation
        for index in range(trg.shape[1]):
            trg_step = torch.index_select(trg, 1, torch.tensor(index).cuda())   # TODO: different mode to use cuda or not

            temp_matrix = torch.empty(batch_size, self.config.en_hidden_size * 2,
                                      self.config.en_hidden_size)  # save the temp matrix for this situation

            attn_input = torch.index_select(encoder_out, 1, torch.tensor(self.config.max_len-1-index).cuda())
            current_degree = torch.index_select(degree, 1, torch.tensor(self.config.max_len-1-index).cuda())
            # temp_matrix = torch.zeros(batch_size, config.en_hidden_size*2, config.de_input_size) # save the temp matrix for this situation
            for index, de in enumerate(current_degree):
                temp_matrix[index] = self.matrix_list[de]

            temp_matrix = temp_matrix.cuda()
            attn_input = F.relu(torch.matmul(attn_input, temp_matrix))

            trg_step = torch.cat([trg_step, attn_input], dim=2)

            out, hidden = self.step_decoding(trg_step, hidden)
            result.append(out)

        out = torch.cat(result, dim=1)
        return out, hidden

    def step_encoding(self, encoder_input, hidden=None):
        output, hidden = self.encoder(encoder_input, hidden)
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

