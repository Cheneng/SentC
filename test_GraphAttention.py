# -*- coding: utf-8 -*-

import torch.optim as optim
from torch.utils.data import DataLoader
from config import GraphAttenConfig
from model import *
from helper import *
from graph_dataset import *
from tqdm import tqdm
import dataset
import pickle
import time
import visdom
import os


def test_GraphAttention():
    DATA_PATH = './data/dataset_eval'
    DICT_PATH = './checkpoint/dict_20000.pkl'
    EMBEDDING_PATH_RANDOM = './model/save_embedding_97and3.ckpt'
    GPU_NUM = 1
    MODEL_PATH = "./checkpoint/Graph_Attn/model-0.ckpt"
    BATCH_SIZE = 500

    torch.cuda.set_device(GPU_NUM)

    embed = get_word_embed().cuda()
    embed_flag = get_flag_embed().cuda()
    vocab = get_vocab()

    config = GraphAttenConfig()
    model = LSTMGraphAttn(config)
    model.load(MODEL_PATH)
    model.cuda()

    trainset = GraphDataset(vocab=vocab, data_path='./data/dataset_eval')
    trainloader = DataLoader(dataset=trainset,
                             batch_size=config.batch_size,
                             collate_fn=graph_fn,
                             pin_memory=True,
                             shuffle=True)

    correct_num = 0
    all_num = 0
    recall_correct = 0
    recall_all = 0

    model.eval()
    for index, (src, trg, neighbor, labels) in enumerate(tqdm(trainloader)):
        print(index)
        batch_size = src.shape[0]
        src = embed(src.cuda())
        trg = embed(trg.cuda())
        neighbor = embed(neighbor.cuda())
        labels = labels.cuda()

        flag4encoder = torch.zeros(src.shape[0], src.shape[1], 3).cuda()
        src = torch.cat([src, flag4encoder], dim=2)

        flag4neighbor = torch.zeros(neighbor.shape[0], neighbor.shape[1], neighbor.shape[2], 3).cuda()
        graph_neighbor = torch.cat([neighbor, flag4neighbor], dim=-1)

        encoder_output, hidden = model.step_encoding(src)   # get the hidden state of the encoder

        output_labels = []
        input_flag = [[2] for j in range(trg.shape[0])]     # 第一位的标志位
        input_flag = torch.Tensor(input_flag).long().cuda()

        output_labels = []
        for index in range(trg.shape[1]):
            flag4decoder = embed_flag(input_flag)
            trg_step = trg[:, index, :].view(batch_size, 1, -1)
            decoder_input = torch.cat([trg_step, flag4decoder], dim=2)  # input是结合attention和当前输入的

            graph_last_input = torch.index_select(encoder_output, 1, torch.tensor(model.config.max_len-1-index).cuda())

            neighbor = torch.index_select(graph_neighbor, 1, torch.tensor(index).cuda())

            graph_attn_out, _ = model.graph_attn(neighbor.squeeze(1), hidden)
            graph_attn_out = graph_attn_out[:, -1, :].view(batch_size, 1, -1)


            # 结合当前的输入和attention得到最终的输出
            trg_step = torch.cat([decoder_input, graph_attn_out], dim=-1)
            trg_step = F.relu(model.linear_out(trg_step))


            out, hidden = model.step_decoding(trg_step, hidden)
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

    P = correct_num / all_num
    R = recall_correct / recall_all
    F1 = 2 * P * R / (P + R)

    print('Finally', BATCH_SIZE)
    print('\tPrecision is {}'.format(P))
    print('\tRecall is {}'.format(R))
    print('\tF1 is {}'.format(F1))
    print(correct_num, recall_correct)
    return P, R, F1


if __name__ == '__main__':
    test_GraphAttention()

