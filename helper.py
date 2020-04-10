import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
import pickle
# from dataset import CompresDataset
from nltk.tokenize import casual_tokenize
import spacy


def graph_fn(batch):
    # The input is [(origin, de_input, neighbor, label)]
    batch_size = len(batch)
    data_batch1 = []
    data_batch2 = []
    data_batch3 = []
    labels_batch = []

    for i in range(batch_size):
        mini_batch = batch[i]
        data_batch1.append(mini_batch[0])
        data_batch2.append(mini_batch[1])
        data_batch3.append(mini_batch[2])
        labels_batch.append(mini_batch[3])

    data_batch1 = padding_sequence(data_batch1, pre_pad=True, max_len=210)
    data_batch2 = padding_sequence(data_batch2, pre_pad=False, max_len=210)
    data_batch3 = padding_sequence(data_batch3, padding_value=[3, 3, 3, 3, 3],
                                   pre_pad=False, max_len=210)
    labels_batch = padding_sequence(labels_batch, pre_pad=False, padding_value=2, max_len=210)

    return torch.LongTensor(data_batch1), torch.LongTensor(data_batch2), \
           torch.LongTensor(data_batch3), torch.LongTensor(labels_batch)


def graph_rank_fn(batch):
    # The input is [(origin, de_input, neighbor, label)]
    batch_size = len(batch)
    data_batch1 = []
    data_batch2 = []
    data_batch3 = []
    labels_batch = []

    for i in range(batch_size):
        mini_batch = batch[i]
        data_batch1.append(mini_batch[0])
        data_batch2.append(mini_batch[1])
        data_batch3.append(mini_batch[2])
        labels_batch.append(mini_batch[3])

    data_batch1 = padding_sequence(data_batch1, pre_pad=True, max_len=210)
    data_batch2 = padding_sequence(data_batch2, pre_pad=False, max_len=210)
    data_batch3 = padding_sequence(data_batch3, pre_pad=False, max_len=210)
    labels_batch = padding_sequence(labels_batch, pre_pad=False, padding_value=2, max_len=210)
    return torch.LongTensor(data_batch1), torch.LongTensor(data_batch2), \
           torch.LongTensor(data_batch3), torch.LongTensor(labels_batch)


def syntax_fn(batch):
    # ((origin, decode, label, pos_tag) ...)
    batch_size = len(batch)
    encode_data = []
    decode_data = []
    labels = []
    tags = []

    for i in range(batch_size):
        mini_batch = batch[i]
        encode_data.append(mini_batch[0])
        decode_data.append(mini_batch[1])
        labels.append(mini_batch[2])
        tags.append(mini_batch[3])

    encode_data = padding_sequence(encode_data, pre_pad=True, max_len=210)
    decode_data = padding_sequence(decode_data, pre_pad=False, max_len=210)
    labels = padding_sequence(labels, pre_pad=False, padding_value=2, max_len=210)
    tags = padding_sequence(tags, pre_pad=False, padding_value=16, max_len=210)

    return torch.LongTensor(encode_data), torch.LongTensor(decode_data), \
           torch.LongTensor(labels), torch.LongTensor(tags)

def my_fn(batch):
    # The input is [(origin, deco_input, label)]
    batch_size = len(batch)
    data_batch1 = []
    data_batch2 = []
    labels_batch = []

    for i in range(batch_size):
        minibatch = batch[i]
        data_batch1.append(minibatch[0])
        data_batch2.append(minibatch[1])
        labels_batch.append(minibatch[2])

    data_batch1 = padding_sequence(data_batch1, pre_pad=False, max_len=210)
    data_batch2 = padding_sequence(data_batch2, pre_pad=False, max_len=210)
    labels_batch = padding_sequence(labels_batch, pre_pad=False, padding_value=2, max_len=210)

    return torch.LongTensor(data_batch1), torch.LongTensor(data_batch2), \
           torch.LongTensor(labels_batch)

def my_parse_fn(batch):
    # The input is [(origin, deco_input, label)]
    batch_size = len(batch)
    data_batch1 = []
    data_batch2 = []    # for parse pos
    data_batch3 = []
    labels_batch = []

    for i in range(batch_size):
        minibatch = batch[i]
        data_batch1.append(minibatch[0])
        data_batch2.append(minibatch[1])
        data_batch3.append(minibatch[2])
        labels_batch.append(minibatch[3])

    data_batch1 = padding_sequence(data_batch1, pre_pad=False, max_len=210)
    data_batch2 = padding_sequence(data_batch2, pre_pad=False, max_len=210)
    data_batch3 = padding_sequence(data_batch3, pre_pad=False, max_len=210)
    labels_batch = padding_sequence(labels_batch, pre_pad=False, padding_value=2, max_len=210)

    return torch.LongTensor(data_batch1), \
           torch.LongTensor(data_batch2), \
           torch.LongTensor(data_batch3), \
           torch.LongTensor(labels_batch)

def my_dis_fn(batch):
    # The input is [(origin, deco_input, label)]
    batch_size = len(batch)
    data_batch1 = []
    data_batch2 = []
    labels_batch = []

    for i in range(batch_size):
        minibatch = batch[i]
        data_batch1.append(minibatch[0])
        data_batch2.append(minibatch[1])
        labels_batch.append(minibatch[2])

    data_batch1 = padding_sequence(data_batch1, pre_pad=True, max_len=210)
    data_batch2 = padding_sequence(data_batch2, pre_pad=True, max_len=210)
    # labels_batch = padding_sequence(labels_batch, pre_pad=False, padding_value=2, max_len=210)

    return torch.LongTensor(data_batch1), torch.LongTensor(data_batch2), \
           torch.LongTensor(labels_batch)

def padding_sequence(sequence, padding_value=0, pre_pad=True, max_len=-1):
    """
        对齐不同长度的句子，方便后面做矩阵运算。

    :param sequence: 输入的list类型句子
    :param padding_value: 补齐使用的数字索引
    :param pre_pad: 是否在句子开头插入补齐
    :return: 对齐了的句子
    """
    new_sent = []

    if max_len == -1:
        # 计算pad最大长度
        for sent in sequence:
            if len(sent) > max_len:
                max_len = len(sent)

    # pad
    for sent in sequence:
        temp = sent

        if len(sent) < max_len:
            if pre_pad is True:
                for i in range(max_len-len(sent)):
                    temp.insert(0, padding_value)
            else:
                for i in range(max_len-len(sent)):
                    temp.append(padding_value)
        else:
            temp = sent[:max_len]
        new_sent.append(temp)
    return new_sent


def get_tag(path='./data/tag_list.pkl'):
    return pickle.load(open(path, 'rb'))


class Vocabulary(object):
    """
    词典类，用来构造词典。
    """
    def __init__(self):
        # 词典到索引 & 索引到词典的索引
        self.word2index = {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2, "<OOV>": 3}
        self.index2word = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3: "<OOV>"}


        # 计算单词出现的词频
        self.count4word = {}
        self.num_words = 4

    # 添加句子中的单词到字典中来
    def add_sentence(self, sentence):
        # for word in sentence.split(' '):
        #     self.add_word(word)
        for word in casual_tokenize(sentence):
            self.add_word(word)

    # 添加单词到字典中来
    def add_word(self, word):
        word = word.lower()
        if word not in self.word2index:
            self.word2index[word] = self.num_words
            self.count4word[word] = 1
            self.index2word[self.num_words] = word
            self.num_words += 1
        else:
            self.count4word[word] += 1

    # 句子中的单词转换成索引
    def sent_to_index(self, sentence):
        index = []
        for word in casual_tokenize(sentence):
            index.append(self.word_to_index(word))
        return index

    def spacy_filter(self, sentence, nlp, pos_dict=None):
        """
            Using spacy to process the sentence. Including tokenize and tag.
        :param sentence: string
        :param nlp: spacy 'en_core_web_sm' object
        :param pos_dict: the position dict
        :return: text_token, pos_out
        """
        text_token = []
        pos_tag = []

        doc = nlp(sentence)
        for token in doc:
            text_token.append(self.word_to_index(token.text))
            pos_tag.append(token.pos_)

        if pos_dict is not None:
            try:
                pos_out = [pos_dict[tag] for tag in pos_tag]
            except:
                raise ValueError('pos dict error', pos_dict)
        else:
            pos_out = pos_tag

        return text_token, pos_out

    # 打印目前词典中的信息
    def print_info(self):
        print('All words: ', len(self.count4word))
        print('dict size: ', len(self.word2index))

    # 使用词频或者是词频出现最高的前n个词的形式来保存字典
    def build_standard_dict(self, saved_by_freq=False,
                            save_freq=50, save_num=80000):
        self.print_info()
        new_count = []
        self.count4word = sorted(self.count4word.items(),
                                 key=lambda x: x[1], reverse=True)
        if saved_by_freq is False:
            # 使用频率最高的前save_num个样本来作为词典
            try:
                new_count = self.count4word[:save_num]
            except:
                raise ValueError('The input save_num is not in [0: dict_size]')
        else:
            for (word, freq) in self.count4word:
                if freq > save_freq:
                    new_count.append((word, freq))

        self.count4word = dict(self.count4word)

        # 根据上面最新的count4word创建新的字典
        self._make_dict(new_count)

    def _make_dict(self, word_freq):
        self.word2index = {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2, "<OOV>": 3}
        self.index2word = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3: "<OOV>"}
        self.num_words = 4
        # 创建新的字典
        for (word, freq) in word_freq:
            self.word2index[word] = self.num_words
            self.index2word[self.num_words] = word
            self.num_words += 1

        self.print_info()

    # 通过词频来对单词进行排序
    def sorted_dict_through_freq(self):
        self.count4word = sorted(self.count4word, key=lambda x:x[1], reverse=True)

    # 单词转换成索引
    def word_to_index(self, word):
        if word not in self.word2index:
            return 3
        else:
            return self.word2index[word]


    def index_to_word(self, index):
        return self.index2word[index]

    def index_list_to_sentence(self, index_list):
        return [self.index_to_word(index) for index in index_list]

    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self, f)



def draw_line(vis, step, lines, names):
    """
        draw multiple lines in one panel.
    :param vis: the object of the visdom.Visdom
    :param step: the step of the line
    :param lines: the lines tuple, (line1, line2, ...)
    :param names: the names tuple, (name1, name2, ...)
    :return: None
    """
    if not len(lines) == len(names):
        raise ValueError('The length of the input is not the same')

    win_name = ''
    for i in range(len(names)):
        win_name += names[i]

    if step == 0:
        for line, name in zip(lines, names):
            vis.line(X=torch.Tensor([step]),
                     Y=torch.Tensor([line]),
                     win=win_name,
                     name='%s' % name,
                     opts=dict(legend=[name])
                     )
    else:
        for line, name in zip(lines, names):
            vis.updateTrace(X=torch.Tensor([step]),
                            Y=torch.Tensor([line]),
                            win=win_name,
                            name='%s' % name
                            )


def get_flag_embed():
    x = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    embed_flag = torch.Tensor(x)
    embed_labels = nn.Embedding.from_pretrained(embed_flag)
    return embed_labels


def get_word_embed(dict_path='./model/save_embedding_97and3.ckpt'):
    embed = nn.Embedding(num_embeddings=20000, embedding_dim=97)
    embed.load_state_dict(torch.load(dict_path))
    return embed


def get_optimizer(optim_fu='adam', model=None, lr=0.01):
    """
        Get the correct optimizer.
    :param optim: which optimizer to choose.{'adam', 'sgd'}
    :param model: the model to optim
    :param lr: the learning rate
    :return: the optimizer in torch.optim
    """
    optimizer_option = {'adam', 'sgd'}
    if optim_fu == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=lr)
    elif optim_fu == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=lr)
    else:
        raise ValueError('The optim should be included in {}'.format(optimizer_option))
    return optimizer


def get_vocab(path='./checkpoint/dict_20000.pkl'):
    vocab = pickle.load(open(path, 'rb'))
    return vocab

def get_gen_trainloader(vocab, batch_size=64, data_path='./data/partial_train_pairs',
                        save_path='./data/dataset.pkl',reload=False):
    print("Loading the train loader.")
    if reload is True:
        data = CompresDataset(vocab=vocab, data_path=data_path)
        with open(save_path, 'wb') as f:
            pickle.dump(data, f)
    else:
        data = pickle.load(open(save_path, 'rb'))
    print("The length of the data is: {}".format(len(data)))
    trainloader = DataLoader(dataset=data,
                             collate_fn=my_fn,
                             batch_size=batch_size,
                             pin_memory=True if torch.cuda.is_available() else False,
                             shuffle=True)
    return trainloader

class Cal_Score(object):
    """
        return P, R, F1, PP
    """
    def __init__(self):
        self.C_rate_remain= 0
        self.C_rate_all = 0
        self.correct_num = 0
        self.batch_num = 0
        self.recall_correct = 0
        self.recall_all = 0
        self.PP = 0
        self.PP_all = 0

    def update(self, preds, labels):
        mask_matrix = (labels < 2)
        # get real preds & labels
        ground_truth = torch.masked_select(labels, mask_matrix)
        predict_labels = torch.masked_select(preds, mask_matrix)

        self.C_rate_all += len(predict_labels)
        self.C_rate_remain += torch.sum(predict_labels).item()

        self.correct_num += torch.sum(predict_labels == ground_truth).item()
        self.batch_num += len(ground_truth)

        p_truth = torch.masked_select(ground_truth, (predict_labels == 1))
        self.PP += torch.sum(p_truth).float()
        self.PP_all += torch.sum(predict_labels)

        self.recall_correct += torch.sum(ground_truth & predict_labels).item()
        self.recall_all += torch.sum(ground_truth).item()

    def print(self):
        P = self.correct_num / self.batch_num
        R = self.recall_correct / self.recall_all
        F1 = 2 * P * R / (P + R)

        prec = self.PP / self.PP_all
        print('Precision {}; Recall {}; F1 {}; pp {}'.format(P, R, F1, prec))

    def reset(self):
        self.C_rate_remain= 0
        self.C_rate_all = 0
        self.correct_num = 0
        self.batch_num = 0
        self.recall_correct = 0
        self.recall_all = 0
        self.PP = 0
        self.PP_all = 0

POS_dict = {'ADJ': 0,
            'ADV': 1,
            'INTJ': 2,
            'NOUN': 3,
            'PROPN': 4,
            'VERB': 5,
            'ADP': 6,
            'AUX': 7,
            'CCONJ': 8,
            'DET': 9,
            'NUM': 10,
            'PART': 11,
            'PRON': 12,
            'SCONJ': 13,
            'PUNCT': 14,
            'SYM': 15,
            'X': 16}

if __name__ == '__main__':
    import os
    import pickle
    from get_embedding import *

    DATA_PATH = './data/data_obj'
    NEW_DATA_PATH = './data/data_num_obj'
    SAVE_PATH = './checkpoint'
    file_name = os.listdir(DATA_PATH)
    file_name = sorted(file_name)
    sent_comp_dict = Vocabulary()

    # create the dict
    for name in file_name:
        print(name)
        with open(os.path.join(DATA_PATH, name), 'rb') as f:
            doc = pickle.load(f)
            for line, headline in zip(doc.origin, doc.headline):
                sent_comp_dict.add_sentence(line)

    sent_comp_dict.build_standard_dict(save_num=20000-4)
    sent_comp_dict.save(SAVE_PATH+'/dict_20000.pkl')
    print(sent_comp_dict.word2index)

