
class LSTM3LayersResConfig(object):
    def __init__(self):
        # training configuration
        self.lr = 0.001
        self.batch_size = 2000
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
        self.num_layers = 1
        self.bidirectional = False
        self.dropout_rate = 0

        # second LSTM configuration
        self.sec_input_size = 100
        self.sec_hidden_size = 100
        self.sec_num_layers = 1
        self.sec_bidirectional = False
        self.sec_dropout_rate = 0

        # output LSTM configuration
        self.last_input_size = 200
        self.last_hidden_size = 100
        self.last_num_layers = 1
        self.last_bidirectional = False
        self.last_dropout_rate = 0
