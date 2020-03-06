import torch
import torch.nn as nn
import torch.nn.functional as F
from .BasicModule import BasicModule
from .Seq2Seq import *


class BiLSTMTSyntaxAttn(BasicModule):
    def __init__(self, config):
        super(BiLSTMTSyntaxAttn, self).__init__()
        self.config = config
        self.encoder = Encoder(config)
        self.decoder = nn.LSTM(input_size=config.de_input_size,
                               hidden_size=config.de_hidden_size,
                               bidirectional=config.de_bidirectional,
                               num_layers=config.de_num_layers,
                               dropout=config.de_dropout_rate,
                               batch_first=True)

        self.syntax_encoder = Encoder(config)
        self.syntax_lstm = nn.LSTM(config.syntax_input_size, config.syntax_output_size, batch_first=True)
        self.linear_syntax = nn.Linear(config.syntax_output_size, config.de_hidden_size)

        self.linear_out = nn.Linear(in_features=2*config.de_hidden_size if config.de_bidirectional else config.de_hidden_size,
                                    out_features=2)

        self.embed = nn.Embedding(17, 97)
        self.embed.weight.requires_grad = False

    def forward(self, src, trg, syntax, decode_flag, hidden=None, syntax_hidden=None):
        encoder_out, hidden = self.step_word_encoding(src, hidden)
        _, hidden_out = self.step_syntax_encoding(syntax, hidden=syntax_hidden)
        syntax_out, _ = self.step_syntax_decoding(syntax, decode_flag, syntax_hidden=hidden_out)

        result = []

        for index in range(trg.shape[1]):
            trg_step = torch.index_select(trg, 1, torch.tensor(index).cuda())   # TODO: different mode to use cuda or not

            attn_input = torch.index_select(encoder_out, 1, torch.tensor(self.config.max_len-1-index).cuda())   # 对应位置的attention
            trg_step = torch.cat([trg_step, attn_input], dim=2)
            # out, hidden = self.attn_step_decoding(trg_step, encoder_out, hidden)
            out, hidden = self.decoder(trg_step, hidden)

            syntax_attn = torch.index_select(syntax_out, 1, torch.tensor(index).cuda())                         # syntax attention
            syntax_attn = F.tanh(self.linear_syntax(syntax_attn))
            attn_out = syntax_attn * out
            attn_out = self.linear_out(attn_out)
            result.append(attn_out)
        out = torch.cat(result, dim=1)
        return out

    def step_word_encoding(self, encoder_input, hidden=None):
        output, hidden = self.encoder(encoder_input, hidden)
        hidden = tuple([torch.index_select(state, 0, torch.tensor(0).cuda()) for state in hidden])
        return output, hidden

    def step_syntax_encoding(self, syntax_input, hidden=None):
        syntax_input = self.embed(syntax_input)
        syntax_flag = torch.zeros(syntax_input.shape[0], syntax_input.shape[1], 3).cuda()
        syntax_input = torch.cat([syntax_input, syntax_flag], dim=-1)
        syntax_out, hidden = self.syntax_encoder(syntax_input, hidden=None)
        hidden_out = [state[1].unsqueeze(0) for state in hidden]    # 选择逆向的那一条
        return syntax_out, hidden_out

    def step_syntax_decoding(self, syntax_trg, decode_flag, syntax_hidden=None):
        if syntax_hidden is None:
            raise ValueError('The hidden state should not be None.')
        syntax = self.embed(syntax_trg)
        syntax = torch.cat([syntax, decode_flag], dim=-1)
        output, hidden = self.syntax_lstm(syntax, syntax_hidden)
        return output, hidden

    def test_decoding(self, decoder_input, syntax, decode_flag, hidden_syntax=None, hidden=None):
        if hidden is None:
            raise ValueError()
        syntax = self.embed(syntax)
        syntax = torch.cat([syntax, decode_flag], dim=-1)
        syntax_out, hidden_syntax = self.syntax_lstm(syntax, hidden_syntax)
        syntax_attn = F.tanh(self.linear_syntax(syntax_out))
        output, hidden = self.decoder(decoder_input, hidden)
        attn_output = syntax_attn * output
        attn_output = self.linear_out(attn_output)
        return attn_output, hidden_syntax, hidden