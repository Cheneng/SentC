import os
import pickle
from SentenceObj import *
from helper import *
import copy
import torch
from torch.utils.data import Dataset
from model import TextRank
from dataset import del_mac_DS, match_list


class GraphDataset(Dataset):

    def __init__(self, vocab, data_path='./data/train_pairs', reverse_src=True):
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
        origin_sentence = self.vocab.sent_to_index(self.origin[index])  # 得到原句并转换成索引
        decoder_input_sentence = copy.deepcopy(origin_sentence)
        decoder_input_sentence.insert(0, 1)  # insert the <sos> to the decoder sentence.

        # compute for the ground truth labels, using the origin <string> type input.
        out_label = match_list(self.origin[index], self.headline[index],
                               flag_index=2)  # flag_index as the label for <sos>

        if self.reverse is True:
            origin_sentence.reverse()

        # To get the graph neighbor of the sentence
        text_ranker = TextRank()
        word_degree = text_ranker.get_degree(self.origin[index])
        word_degree.insert(0, 0)

        return origin_sentence, decoder_input_sentence, word_degree, out_label

    def __len__(self):
        return len(self.origin)

if __name__ == '__main__':
    DICT_PATH = './checkpoint/dict_20000.pkl'
    # vocabulary类
    vocab = pickle.load(open(DICT_PATH, 'rb'))
    data = GraphDataset(vocab, reverse_src=True)
    origin, decode, degree, label = data[9]

    print(origin)
    print(decode)
    print(degree)
    print(label)