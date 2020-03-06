import os
from model import SemanticDiscriminator
from config import SeDiscriminatorConfig
from helper import *
import dataset
import pickle
from nltk import pos_tag
from nltk.tokenize import casual_tokenize
from model import LSTMGraphAttn
from config import GraphAttenConfig

config = GraphAttenConfig()
model = LSTMGraphAttn(config)



src = torch.randn(10, 5, 100)
trg = torch.randn(10, 6, 100)
neighbor = torch.randn(10, 6, 5, 97)

input = model(src, trg, neighbor)
print(input)

# data_path = './data/train_pairs'
# save_path = './data/GAN_pair/real_data'
# dis_model_path = './checkpoint/GAN_train/discriminator/pretrain_epoch0'
#
# TAG = ['ADJ', 'ADP', 'ADV', 'CONJ', 'DET',
#        'NOUN', 'NUM', 'PRT', 'PRON', 'VERB', '.', 'X']
# TAG_DICT = {item: num for num, item in enumerate(TAG)}
# print(TAG_DICT)
#
# x = "chencheng is a very good person, and any one loves him. In 2018 century !"
# y = casual_tokenize(x)
# print(y)
# x = pos_tag(y, tagset='universal')
#
# output = []
# for _, item in x:
#     output.append(TAG_DICT[item])
# print(output)

# # test discriminator
# dis_config = SeDiscriminatorConfig()
# dis_model = SemanticDiscriminator(dis_config)
# dis_model.load(dis_model_path)
#
#
# dis_model.load(dis_model_path)
# embed = get_word_embed()
#
# save_path='./data/dataset.pkl'
# dataset = pickle.load(open(save_path, 'rb'))


