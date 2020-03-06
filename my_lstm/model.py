import torch
import torch.nn as nn
import torch.nn.functional as F
from .BasicModule import BasicModule

class MyLSTM(BasicModule):
    def __init__(self, input_dim, tag_dim, hidden_dim):
        super(BasicModule, self).__init__()
        self.input_dim = input_dim
        self.tag_dim = tag_dim
        self.hidden_dim = hidden_dim

        self.Wf = nn.Parameter(torch.Tensor(input_dim, hidden_dim))     # forget gate
        self.Uf = nn.Parameter(torch.Tensor(hidden_dim, hidden_dim))
        self.bf = nn.Parameter(torch.Tensor(hidden_dim))

        self.Wi = nn.Parameter(torch.Tensor(input_dim, hidden_dim))     # input gate
        self.Ui = nn.Parameter(torch.Tensor(hidden_dim, hidden_dim))
        self.bi = nn.Parameter(torch.Tensor(hidden_dim))

        self.Wx = nn.Parameter(torch.Tensor(input_dim, hidden_dim))     # info cell
        self.Ux = nn.Parameter(torch.Tensor(hidden_dim, hidden_dim))
        self.bx = nn.Parameter(torch.Tensor(hidden))

        self.Wo = nn.Parameter(torch.Tensor(tag_dim, hidden_dim))      # output gate
        self.Uo = nn.Parameter(torch.Tensor(hidden_dim, hidden_dim))
        self.bo = nn.Parameter(torch.Tensor(tag_dim))


    def forward(self, x, tag, hidden=None):
        b_size = x.size(0)
        if hidden is None:
            hidden = torch.zeros(1, b_size, self.hidden_dim)

        # forget gate
        f = F.sigmoid(
            torch.matmul(x, self.Wf) +
            torch.matmul(hidden, self.Wf) +
            self.bf
        )

        # input gate
        i = F.sigmoid(
            torch.matmul(x, self.Wi) +
            torch.matmul(hidden, self.Ui) +
            self.bi
        )

        # output gate
        o = F.sigmoid(
            torch.matmul(x, self.Wo) +
            torch.matmul(hidden, self.Ui) +
            self.bo
        )

        # info memory
        info = F.tanh(
            torch.matmul(tag, self.Wx) +
            torch.matmul(hidden, self.Ux) +
            self.bx
        )




