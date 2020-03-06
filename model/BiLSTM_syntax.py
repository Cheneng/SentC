import torch
import torch.nn as nn
import torch.nn.functional as F
from .BasicModule import BasicModule
from .Seq2Seq import *


class BiLSTMSyntax(BasicModule):
    def __init__(self, config):
        super(BiLSTMSyntax, self).__init__()
        self.config = config
        self.encoder = Encoder(config)
        self.decoder = nn.LSTM(config.de_input_size, config.de_hidden_size, batch_first=True)
        self.syntax_encoder = Encoder(config)
        self.syntax_lstm = nn.LSTM(config.syntax_input, config.syntax_output, batch_first=True)
        self.linear_syntax = nn.Linear(config.syntax_output, config.de_hidden_size)
        self.linear_out = nn.Linear(config.de_hidden_size, 2)
        self.embed = nn.Embedding(17, 97)   # the syntax embedding
        self.embed.weight.requires_grad = False

    def forward(self, src, trg, syntax, decode_flag, hidden=None):
        _, encoder_hidden, syntax_hidden = self.step_encoding(src, syntex_input=syntax, hidden=hidden)     # encoder获取句子embedding
        out, hidden = self.step_decoding(trg, syntax, decode_flag, hidden=encoder_hidden, hidden_syntax=syntax_hidden)
        return out, hidden

    def step_encoding(self, encoder_input, syntex_input, hidden=None):
        output, hidden = self.encoder(encoder_input, hidden)
        hidden = tuple([torch.index_select(state, 0, torch.tensor(0).cuda()) for state in hidden])  # 选择前向的lstm
        syntex_input = self.embed(syntex_input)
        syntex_flag = torch.zeros(syntex_input.shape[0], syntex_input.shape[1], 3).cuda()
        syntex_input = torch.cat([syntex_input, syntex_flag], dim=-1)
        _, syntex_hidden = self.syntax_encoder(syntex_input)
        syntex_hidden = tuple([torch.index_select(state, 0, torch.tensor(1).cuda()) for state in syntex_hidden])  # 选择后向的lstm
        return output, hidden, syntex_hidden

    def step_decoding(self, decoder_input, syntax, decode_flag, hidden_syntax=None, hidden=None):
        if hidden is None:
            raise ValueError("The hidden should not be None")
        syntax = self.embed(syntax)
        syntax = torch.cat([syntax, decode_flag], dim=-1)
        syntax_out, _ = self.syntax_lstm(syntax, hidden_syntax)
        syntax_attn = self.linear_syntax(syntax_out)    # 捕获语义信息
        syntax_attn = F.tanh(syntax_attn)
        output, hidden = self.decoder(decoder_input, hidden)    # decoder出结果
        attn_output = syntax_attn * output                      # 通过门控限制
        attn_output = self.linear_out(attn_output)
        return attn_output, hidden

    def test_decoding(self, decoder_input, syntax, decode_flag, hidden_syntax=None, hidden=None):
        if hidden is None:
            raise ValueError("The hidden should not be None")
        syntax = self.embed(syntax)
        syntax = torch.cat([syntax, decode_flag], dim=-1)
        syntax_out, hidden_syntax = self.syntax_lstm(syntax, hidden_syntax)
        syntax_attn = F.tanh(self.linear_syntax(syntax_out))    # 捕获语义信息
        output, hidden = self.decoder(decoder_input, hidden)    # decoder出结果
        attn_output = syntax_attn * output                      # 通过门控限制
        attn_output = self.linear_out(attn_output)
        return attn_output, hidden_syntax, hidden

