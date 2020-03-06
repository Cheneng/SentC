import torch
from torch.utils.data import DataLoader
from dataset_add_info import CompresDataset
from helper import get_vocab, get_tag
from nltk import pos_tag
from nltk.tokenize import casual_tokenize
import pickle

TAG_DICT = {'ADJ': 0, 'ADP': 1, 'ADV': 2, 'CONJ': 3, 'DET': 4, 'NOUN': 5,
            'NUM': 6, 'PRT': 7, 'PRON': 8, 'VERB': 9, '.': 10, 'X': 11}

def tagger(str):
    word_list = casual_tokenize(str)
    tag_list = pos_tag(word_list, tagset='universal')
    return tag_list

def str2int_list(str, split_flag=' '):
    str_list = str.split(split_flag)
    int_list = map(int, str_list)
    return list(int_list)

vocab = get_vocab()
data = CompresDataset(vocab=vocab)
dataloader = DataLoader(data, batch_size=1)
print(len(data))

x = get_tag()
print(x[:10])

#
# for step, (origin, headline, label) in enumerate(dataloader, 1):
#     out_tag = tagger(origin[0])
#     temp_list = []
#     print(step)
#     for item in out_tag:
#         temp_list.append(TAG_DICT[item[1]])
#     tag_list.append(temp_list)
#
# with open('./data/tag_list.pkl', 'wb') as f:
#     pickle.dump(tag_list, f)

