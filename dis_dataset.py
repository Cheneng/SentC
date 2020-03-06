import torch
import os
import random
import pickle
import torch.utils.data as data
from dataset import del_mac_DS

class DisDataset(data.Dataset):
    def __init__(self, vocab, data_path='./data/train_pairs', negative=False):
        self.negative = negative
        self.data_path = data_path
        self.file_name = del_mac_DS(os.listdir(self.data_path))
        self.vocab = vocab

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
        origin_sentence = self.vocab.sent_to_index(self.origin[index])
        headline_sentence = self.vocab.sent_to_index(self.headline[index])
        label = [1]
        if self.negative:
            del_num = random.randint(1, 3)
            for i in range(del_num):
                length = len(headline_sentence)
                del_index = random.randint(0, length-1)
                headline_sentence.remove(headline_sentence[del_index])  # delete
            label = [0]
        return origin_sentence, headline_sentence, label

    def __len__(self):
        return len(self.origin)


if __name__ == '__main__':
    save_path = './checkpoint/dict_20000.pkl'
    vocab = pickle.load(open(save_path, 'rb'))
    x = DisDataset(vocab=vocab)
    y = DisDataset(vocab=vocab, negative=True)
    print(len(x))
    print(len(y))

    o, h, label = x[10]
    o_, h_, label_ = y[10]

    print(o)
    print(h)
    print(label)

    print(o_)
    print(h_)
    print(label_)