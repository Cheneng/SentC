import torch
import torch.nn as nn
import torch.nn.functional as F
from .BasicModule import BasicModule


class Encoder(BasicModule):
    def __init__(self, config):
        super(Encoder, self).__init__()
        self.encoder = nn.LSTM(input_size=config.en_input_size,
                               hidden_size=config.en_hidden_size,
                               bidirectional=config.en_bidirectional,
                               num_layers=config.en_num_layers,
                               dropout=config.en_dropout_rate,
                               batch_first=True)

    def forward(self, x, hidden=None):
        output, hidden_out = self.encoder(x, hidden)
        return output, hidden_out

class Decoder(BasicModule):
    def __init__(self, config):
        super(Decoder, self).__init__()
        self.decoder = nn.LSTM(input_size=config.de_input_size,
                               hidden_size=config.de_hidden_size,
                               bidirectional=config.de_bidirectional,
                               num_layers=config.de_num_layers,
                               dropout=config.de_dropout_rate,
                               batch_first=True)
        self.linear_out = nn.Linear(in_features=2*config.de_hidden_size if config.de_bidirectional else config.de_hidden_size,
                                    out_features=2)

    def forward(self, x, hidden=None):
        output, hidden = self.decoder(x, hidden)
        output = self.linear_out(output)
        return output, hidden


class optDecoder(BasicModule):
    def __init__(self, config):
        super(optDecoder, self).__init__()
        self.decoder = nn.LSTM(input_size=config.de_input_size,
                               hidden_size=config.de_hidden_size,
                               bidirectional=config.de_bidirectional,
                               num_layers=config.de_num_layers,
                               dropout=config.de_dropout_rate,
                               batch_first=True)
        self.linear_out = nn.Linear(in_features=2*config.de_hidden_size if config.de_bidirectional else config.de_hidden_size,
                                    out_features=config.out_feature_num)

    def forward(self, x, hidden=None):
        output, hidden = self.decoder(x, hidden)
        output = self.linear_out(output)
        output = F.log_softmax(output, dim=1)
        return output, hidden


class Seq2Seq(BasicModule):
    def __init__(self, config):
        super(Seq2Seq, self).__init__()
        self.encoder = Encoder(config)
        self.decoder = Decoder(config)

    def forward(self, src, trg, hidden=None):
        _, encoder_hidden = self.encoder(src, hidden)
        out, hidden = self.decoder(trg, hidden=encoder_hidden)
        return out, hidden

    def step_decoding(self, decoder_input, hidden=None):
        if hidden is None:
            raise ValueError("The hidden should not be None")
        output, hidden = self.decoder(decoder_input, hidden)
        return output, hidden

    def step_encoding(self, encoder_input, hidden=None):
        output, hidden = self.encoder(encoder_input, hidden)
        return output, hidden


class SeqAutoEncoder(BasicModule):
    def __init__(self, config):
        super(SeqAutoEncoder, self).__init__()
        self.encoder = Encoder(config)
        self.decoder = optDecoder(config)

    def forward(self, src, trg, hidden=None):
        _, encoder_hidden = self.encoder(src, hidden)
        out, hidden = self.decoder(trg, hidden=encoder_hidden)
        return out, hidden

    def step_decoding(self, decoder_input, hidden=None):
        if hidden is None:
            raise ValueError("The hidden should not be None")
        output, hidden = self.decoder(decoder_input, hidden)
        return output, hidden

    def step_encoding(self, encoder_input, hidden=None):
        output, hidden = self.encoder(encoder_input, hidden)
        output = F.softmax(output)
        return output, hidden


if __name__ == '__main__':
    from config import Config
    config = Config()
    model = Seq2Seq(config)
    x = torch.rand(2, 3, 100)
    out, hidden = model(x, x)
    print(out)

