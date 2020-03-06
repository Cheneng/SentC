
class LSTM3LayersConfig(object):
    def __init__(self):
        # training configuration
        self.lr = 0.001
        self.batch_size = 512
        self.epoch = 5
        self.batch_first = True

        # dataset configuration
        self.dict_size = 20000
        self.pad_flag_index = 0     # padding在字典中的索引
        self.sos_flag_index = 1       # 开始标志在字典中的索引
        self.eos_flag_index = 2
        self.oov_flag_index = 3         # 超出字典词在字典中的索引

        # LSTM configuration
        self.input_size = 100
        self.hidden_size = 100
        self.num_layers = 3
        self.bidirectional = False
        self.dropout_rate = 0.5


