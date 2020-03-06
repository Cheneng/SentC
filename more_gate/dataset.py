import torch
import os
import torch.utils.data as data
import pickle
from SentenceObj import *
from helper import *
import copy
from nltk.tokenize import casual_tokenize
from helper import syntax_fn

FLAG_SOS = 1


class CompresDataset(data.Dataset):
    """
    Dataset for origin sentence and headline.
    """
    def __init__(self, vocab, data_path='./data/train_pairs', reverse_src=False):
        """
            用来做输入的数据集。
        :param vocab: 包括将词转索引的功能类
        :param data_path: 输入文件所在地址
        :param reverse_src: 是否翻转输入
        """
        self.data_path = data_path
        self.file_name = del_mac_DS(os.listdir(self.data_path))
        self.vocab = vocab
        self.reverse = reverse_src

        self.origin = []
        self.headline = []

        for name in self.file_name:
            print('loading', name)
            with open(os.path.join(self.data_path, name), 'r') as f:
                for line in f:
                    line = line.split('\t', 1)
                    self.origin.append(line[0])
                    self.headline.append(line[1])

    def __getitem__(self, index):
        """
        Return:
            1. 原始句子作为encoder输入
            2. 带有开头<SOS>的句子作为decoder输入
            3. 输出label的ground truth
        """

        origin_sentence = self.vocab.sent_to_index(self.origin[index])
        # headline_sentence = self.vocab.sent_to_index(self.headline[index])
        decode_input_sentence = copy.deepcopy(origin_sentence)
        decode_input_sentence.insert(0, 1)  # insert the <sos> to the decoder sentence.

        # compute for the ground truth labels, using the origin <string> type input.
        out_label = match_list(self.origin[index], self.headline[index],
                               flag_index=2)  # flag_index as the label for <sos>

        if self.reverse is True:
            origin_sentence.reverse()

        # return self.origin[index], self.headline[index], out_label    # test for origin sentence
        return origin_sentence, decode_input_sentence, out_label

    def __len__(self):
        return len(self.origin)


class SeqAutoEncoderDataset(data.Dataset):
    """
    Dataset for encoder and decoder.
    """
    def __init__(self, vocab, data_path='./data/train_pairs', reverse_src=False):
        """
            用来做输入的数据集。
        :param vocab: 包括将词转索引的功能类
        :param data_path: 输入文件所在地址
        :param reverse_src: 是否翻转输入
        """
        self.data_path = data_path
        self.file_name = del_mac_DS(os.listdir(self.data_path))
        self.vocab = vocab
        self.reverse = reverse_src

        self.sentence = []

        for name in self.file_name:
            print('loading', name)
            with open(os.path.join(self.data_path, name), 'r') as f:
                for line in f:
                    line = line.split('\t', 1)
                    self.sentence.append(line[0])
                    self.sentence.append(line[1])

    def __getitem__(self, index):
        """
        Return:
            1. 句子
            2. 带有开头<SOS>的句子作为decoder输入
            3. 输出label的ground truth
        """
        origin_sentence = self.vocab.sent_to_index(self.sentence[index])
        decode_input_sentence = copy.deepcopy(origin_sentence)
        decode_input_sentence.insert(0, 1)  # insert the <sos> to the decoder sentence.
        decode_input_sentence.pop()

        if self.reverse is True:
            origin_sentence.reverse()

        origin_sentence = padding_sequence([origin_sentence], pre_pad=True, max_len=210, padding_value=0)
        decode_input_sentence = padding_sequence([decode_input_sentence], pre_pad=False, max_len=210, padding_value=0)

        origin_sentence = torch.Tensor(origin_sentence).long()
        decode_input_sentence = torch.Tensor(decode_input_sentence).long()

        return origin_sentence, decode_input_sentence

    def __len__(self):
        return len(self.sentence)


class SyntaxDataset(data.Dataset):
    def __init__(self, vocab, data_path='../data/train_pairs', reverse_src=False):
        self.data_path = data_path
        self.file_name = del_mac_DS(os.listdir(self.data_path))
        self.vocab = vocab
        self.reverse = reverse_src
        self.nlp = spacy.load('en_core_web_sm')
        self.origin = []
        self.headline = []
        self.pos_dict = POS_dict

        for name in self.file_name:
            print('loading', name)
            with open(os.path.join(self.data_path, name), 'r') as f:
                for line in f:
                    line = line.split('\t', 1)
                    self.origin.append(line[0])
                    self.headline.append(line[1])

    def __getitem__(self, index):
        origin_sentence, pos = self.vocab.spacy_filter(self.origin[index], self.nlp, self.pos_dict)
        decode_input_sentence = copy.deepcopy(origin_sentence)
        decode_input_sentence.insert(0, 1)
        pos.insert(0, 16)
        out_label = match_list(self.origin[index], self.headline[index],
                               flag_index=2)    # flag_index as the label for <sos>
        if self.reverse is True:
            origin_sentence.reverse()

        return origin_sentence, decode_input_sentence, out_label, pos

    def __len__(self):
        return len(self.origin)


def match_list(origin, headline, flag_for_sentence=True, flag_index=0):
    """
        用来获得字符匹配的ground_truth

    :param origin: 原句子
    :param headline: 压缩句子
    :param flag_for_sentence: 是否插入表示句子开头结尾的标志
    :param flag_index: 插入开头标志对应标签
    :return: 原句子是否留下来的标签
    """

    origin = split_word(origin)
    headline = split_word(headline)

    out_list = []
    index_head = 0
    for orig in origin:
        if index_head < len(headline) and headline[index_head] == orig:
            out_list.append(1)
            index_head += 1
        else:
            out_list.append(0)

    assert len(out_list) == len(origin), 'dimension not match #1{0}, #2{1}'.format(len(out_list), len(origin))

    # 判定是否插入句子开始结束标志
    if flag_for_sentence is True:
        out_list.insert(0, flag_index)
    return out_list


def del_mac_DS(list_name, MAC_EXTRAL='.DS_Store'):
    if MAC_EXTRAL in list_name:
        list_name.remove(MAC_EXTRAL)
    list_name = sorted(list_name)
    return list_name


def split_word(sentence):
    return_list = []
    for word in casual_tokenize(sentence):
        return_list.append(word.lower())
    return return_list


if __name__ == '__main__':
    save_path = './checkpoint/dict_20000.pkl'
    vocab = pickle.load(open(save_path, 'rb'))
    x = SyntaxDataset(vocab=vocab)
    train_loader = data.DataLoader(x, collate_fn=syntax_fn,
                                   batch_size=2)

    print(len(x))

    # def print_pairs(data):
    #     o, t, l, p = data
    #     print(o)
    #     print(t)
    #     print(l)
    #     print(p)
    #
    # print_pairs(x[50000])

    for data1, data2, data3, data4 in train_loader:
        print(data3)
        print(data2)
