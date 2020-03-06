
class SeDiscriminatorConfig(object):
    def __init__(self):
        self.lr = 0.001
        self.batch_size = 256
        self.input_dim = 100

        self.hidden_dim = 100
        self.num_layers = 1
        self.bidirectional = True

        self.dropout_rate = 0.75
        self.out_class = 2
        self.epoch = 1