import torch
import torch.nn as nn
import torch
import torch.nn as nn
import torch.nn.functional as F
from .Seq2Seq import *
from .BasicModule import BasicModule


class LSTM3LayersResSyntax(BasicModule):
    def __init__(self, config):
        super(LSTM3LayersResSyntax, self).__init__()
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


        self.syntax_encoder = Encoder(config)
        self.syntax_lstm = nn.LSTM(config.syntax_input_size, config.syntax_output_size, batch_first=True)
        self.linear_syntax = nn.Linear(config.syntax_output_size, config.last_hidden_size)

        self.linear_out = nn.Linear(in_features=2*config.last_hidden_size if config.last_bidirectional else config.last_hidden_size,
                                    out_features=2)

        self.embed = nn.Embedding(17, 97)
        self.embed.weight.requires_grad = False

    def forward(self, src, trg, syntax, decode_flag, hidden=None, syntax_hidden=None):
        _, word_hidden = self.step_word_encoding(src, hidden=hidden)
        _, syntax_hidden = self.step_syntax_encoding(syntax, hidden=syntax_hidden)
        word_out, _ = self.step_word_decoding(trg, hidden=word_hidden)
        syntax_out, _ = self.step_syntax_decoding(syntax, decode_flag, syntax_hidden=syntax_hidden)
        syntax_attn = self.linear_syntax(syntax_out)
        syntax_attn = F.tanh(syntax_attn)
        attn_out = syntax_attn * word_out
        attn_out = self.linear_out(attn_out)
        return attn_out


    def step_word_encoding(self, src, hidden=None):
        output1, hidden1 = self.lstm1(src, hidden)
        output2, hidden2 = self.lstm2(output1, hidden)
        last_src = torch.cat([output2, src], dim=2)
        output, hidden3 = self.lstm3(last_src, hidden)
        hidden = [hidden1, hidden2, hidden3]
        return output, hidden

    def step_syntax_encoding(self, syntax_input, hidden=None):
        syntax_input = self.embed(syntax_input)
        syntax_flag = torch.zeros(syntax_input.shape[0], syntax_input.shape[1], 3).cuda()
        syntax_input = torch.cat([syntax_input, syntax_flag], dim=-1)
        syntax_out, hidden = self.syntax_encoder(syntax_input, hidden=None)
        hidden_out = [state[0].unsqueeze(0) for state in hidden]
        return syntax_out, hidden_out

    def step_word_decoding(self, trg, hidden=None):
        if hidden is None:
            raise ValueError('The hidden state should not be None.')
        output1, hidden1 = self.lstm1(trg, hidden[0])
        output2, hidden2 = self.lstm2(output1, hidden[1])
        last_trg = torch.cat([output2, output1], dim=-1)
        output3, hidden3 = self.lstm3(last_trg, hidden[2])
        hidden = [hidden1, hidden2, hidden3]
        return output3, hidden

    def step_syntax_decoding(self, syntax, decode_flag, syntax_hidden=None):
        if syntax_hidden is None:
            raise ValueError('The hidden state should not be None.')
        syntax = self.embed(syntax)
        syntax = torch.cat([syntax, decode_flag], dim=-1)
        output, hidden = self.syntax_lstm(syntax, syntax_hidden)
        return output, hidden

    def test_decoding(self, decoder_input, syntax, decode_flag, hidden_syntax=None, hidden=None):
        if hidden is None:
            raise ValueError()
        syntax = self.embed(syntax)
        syntax = torch.cat([syntax, decode_flag], dim=-1)
        syntax_out, hidden_syntax = self.syntax_lstm(syntax, hidden_syntax)
        syntax_attn = F.tanh(self.linear_syntax(syntax_out))    # attention gate
        output, hidden = self.step_word_decoding(decoder_input, hidden)
        attn_output = syntax_attn * output
        attn_output = self.linear_out(attn_output)
        return attn_output, hidden_syntax, hidden

    #
    # def step_decoding(self, src, hidden=None):
    #     if hidden is None:
    #         raise ValueError("The decoding hidden should not be None")
    #     output1, hidden1 = self.lstm1(src, hidden[0])
    #     output2, hidden2 = self.lstm2(output1, hidden[1])
    #     last_src = torch.cat([output2, src], dim=2)
    #     output3, hidden3 = self.lstm3(last_src, hidden[2])
    #     output = self.linear_out(output3)
    #     hidden = [hidden1, hidden2, hidden3]
    #     return output, hidden

