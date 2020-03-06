# -*- coding: utf-8 -*-

import torch.optim as optim
from torch.utils.data import DataLoader
from config import *
from model import *
from helper import *
import dataset
import pickle
import time
import visdom
import os


def test_3layerLSTM(model_num=1):
    # DATA_PATH = './data/dataset_eval'
    DATA_PATH = './nbc'

    DICT_PATH = './checkpoint/dict_20000.pkl'
    EMBEDDING_PATH_RANDOM = './model/save_embedding_97and3.ckpt'
    GPU_NUM = 0
    MODEL_PATH = './checkpoint/LSTM3Layers/model{}.ckpt'
    BATCH_SIZE = 100

    torch.cuda.set_device(GPU_NUM)

    # vocabulary类
    vocab = pickle.load(open(DICT_PATH, 'rb'))

    testset = dataset.CompresDataset(vocab=vocab, data_path=DATA_PATH)
    testloader = DataLoader(dataset=testset,
                            collate_fn=my_fn,
                            batch_size=BATCH_SIZE,
                            pin_memory=True)

    config = LSTM3LayersConfig()
    model = LSTM3Layers(config).cuda()

    model.load(MODEL_PATH.format(model_num))

    # 预训练好的词向量读取
    embed = nn.Embedding(num_embeddings=20000, embedding_dim=97)
    embed.load_state_dict(torch.load(EMBEDDING_PATH_RANDOM))
    embed.cuda()

    embed_flag = get_flag_embed().cuda()

    correct_num = 0
    all_num = 0

    recall_correct = 0
    recall_all = 0

    model.eval()
    for i, (src, trg, labels) in enumerate(testloader):
        print(i*src.shape[0])
        src = embed(src.cuda())
        trg = trg.cuda()
        labels = labels.cuda()
        # 添加三位全0向量给encoder输入
        flag4encoder = torch.zeros(src.shape[0], src.shape[1], 3).cuda()
        src = torch.cat([src, flag4encoder], dim=2)

        _, hidden = model.step_encoding(src)    # get the encoder hidden state
        hidden = tuple([state.view(config.num_layers, -1, config.hidden_size) for state in hidden])

        output_labels = []
        input_flag = [[2] for j in range(trg.shape[0])]
        input_flag = torch.Tensor(input_flag).long().cuda()
        for index in range(trg.shape[1]):
            # Prepare for the input
            flag4encoder = embed_flag(input_flag)
            select_elem = torch.index_select(trg, 1, torch.tensor(index).cuda())
            decoder_input = embed(select_elem)
            decoder_input = torch.cat([decoder_input, flag4encoder], dim=2)

            out, hidden = model.step_decoding(decoder_input, hidden)
            input_flag = torch.max(out, 2)[1]
            output_labels.append(input_flag)

        output_labels = torch.cat(output_labels, dim=1)
        labels = labels.squeeze()
        mask_matrix = labels < 2
        predict_labels = torch.masked_select(output_labels, mask_matrix)
        ground_truth = torch.masked_select(labels, mask_matrix)

        correct_num += torch.sum(predict_labels == ground_truth).item()
        recall_correct += torch.sum(predict_labels & ground_truth).item()
        recall_all += torch.sum(ground_truth).item()
        all_num += len(ground_truth)

        P = correct_num / all_num
        R = recall_correct / recall_all
        F1 = 2 * P * R / (P + R)
        print('Precision is {}'.format(P))
        print('Recall is {}'.format(R))
        print('F1 is {} \n'.format(F1))
        print('Compression Rate is {}'.format(recall_correct / all_num))

    P = correct_num / all_num
    R = recall_correct / recall_all
    F1 = 2 * P * R / (P + R)
    print('Finally')
    print('\tPrecision is {}'.format(P))
    print('\tRecall is {}'.format(R))
    print('\tF1 is {}'.format(F1))
    print('Compression Rate is {}'.format(recall_correct / all_num))
    return P, R, F1


if __name__ == '__main__':
    test_3layerLSTM(3)
