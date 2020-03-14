import torch
import torch.nn as nn
import torch.nn.functional as F
import math
try:
    from .BasicModule import BasicModule
except:
    from BasicModule import BasicModule


class BasicTransformer(BasicModule):
    def __init__(self, d_model=100, nhead=4, num_encoder_layer=4,
                 num_decoder_layer=4, dim_feedforward=400):
        super(BasicTransformer, self).__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, 
                                                   nhead=nhead,
                                                   dim_feedforward=dim_feedforward)
        encoder_norm = nn.LayerNorm(d_model)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layer, norm=encoder_norm)

        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, 
                                                  nhead=nhead,
                                                  dim_feedforward=dim_feedforward)
        decoder_norm = nn.LayerNorm(d_model)
        # decoder_norm = None
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layer, norm=decoder_norm)

        self.out_linear = nn.Linear(d_model, 2)

        
    def forward(self, src, tgt, src_key_padding_mask=None, tgt_key_padding_mask=None, 
                memory_key_padding_mask=None, tgt_mask=None):
        memory = self.encode(src, src_key_padding_mask=src_key_padding_mask)
        decode_out = self.decode(tgt, memory, tgt_mask=tgt_mask, 
                                 tgt_key_padding_mask=tgt_key_padding_mask,
                                 memory_key_padding_mask=memory_key_padding_mask)
        out = self.out_linear(decode_out)
        return out

    def encode(self, src, src_key_padding_mask=None):
        memory = self.encoder(src, src_key_padding_mask=src_key_padding_mask)
        return memory

    def decode(self, tgt, memory, tgt_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None):
        out = self.decoder(tgt, memory, tgt_mask=tgt_mask, 
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask)
        return out
    
    def decode_last(self, tgt, memory, tgt_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None):
        out = self.decode(tgt, memory, tgt_mask=tgt_mask, 
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask)
        out = self.out_linear(out[-1])
        return out


    def generate_square_subsequent_mask(self, sz):
        r"""Generate a square mask for the sequence. The masked positions are filled with float('-inf').
            Unmasked positions are filled with float(0.0).
        """
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        if next(self.parameters()).is_cuda is True:
            mask = mask.cuda()
        return mask

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

class PositionalEncoding(nn.Module):
    
    def __init__(self, d_model, dropout=0.1, max_len=210):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        # print('div_term', div_term.size())
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 0:
            pe[:, 1::2] = torch.cos(position * div_term)
        else:
            # print('pe2', pe[:, 1::2].size())
            # print('cos', position.size())
            pe[:, 1::2] = torch.cos(position * div_term[:-1])


        # pe = pe.unsqueeze(0).transpose(0, 1)
        pe = pe.unsqueeze(0)        # [batch_size, seq_len, embedding]
        self.register_buffer('pe', pe)
        print('pe', pe.size())

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


if __name__ == '__main__':
    model = BasicTransformer()
    src = torch.randn(20, 1, 100)
    trg = torch.randn(30, 1, 100)
    out = model(src, trg)

    mask = model.generate_square_subsequent_mask(30)
    out = model(src, trg, tgt_mask=mask)

    # print(out.size())
    memory = model.encode(src)
    print('memory', memory.size())

    out = model.decode(trg, memory)

    print('out', out.size())
    print('last out', out[-1].size())

    out = model.decode_last(trg, memory)
    print(torch.max(out, -1)[1].size())
    print('decode_last', out.size())


    # position_model = PositionalEncoding(d_model=97)
    # p = position_model(torch.randn(20, 16, 97))
    # print(p.size())

    # print(next(model.parameters()).is_cuda)
    # print(model.parameters)


