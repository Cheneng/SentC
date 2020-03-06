# -*- coding: utf-8 -*-

from torch.utils.data import DataLoader
from config import BiLSTMTAttnSyntaxConfig
from model import BiLSTMTSyntaxAttn
from helper import *
from dataset import SyntaxDataset
from tqdm import tqdm
import dataset
import pickle
import time
import visdom
import os


def test_GraphAttention():
    # DATA_PATH = './data/dataset_eval'
    DATA_PATH = './nbc'

    DICT_PATH = './checkpoint/dict_20000.pkl'
    EMBEDDING_PATH_RANDOM = './model/save_embedding_97and3.ckpt'
    GPU_NUM = 0
    model_epoch = 10
    MODEL_PATH = f"./checkpoint/syntax_t_Attn/model-{model_epoch}.ckpt"
    BATCH_SIZE = 200

    torch.cuda.set_device(GPU_NUM)

    embed = get_word_embed().cuda()
    embed_flag = get_flag_embed().cuda()
    vocab = get_vocab()

    config = BiLSTMTAttnSyntaxConfig()
    model = BiLSTMTSyntaxAttn(config)
    model.load(MODEL_PATH)
    model.cuda()

    trainset = SyntaxDataset(vocab=vocab, data_path=DATA_PATH)
    trainloader = DataLoader(dataset=trainset,
                             batch_size=config.batch_size,
                             collate_fn=syntax_fn,
                             num_workers=2,
                             pin_memory=True,
                             shuffle=False)

    correct_num = 0
    all_num = 0
    recall_correct = 0
    recall_all = 0

    model.eval()
    for index, (src, trg, labels, tags) in enumerate(tqdm(trainloader)):
        print(index)
        batch_size = src.shape[0]
        src = embed(src.cuda())
        trg = embed(trg.cuda())
        labels = labels.cuda()
        tags = tags.cuda()

        flag4encoder = torch.zeros(src.shape[0], src.shape[1], 3).cuda()
        src = torch.cat([src, flag4encoder], dim=2)

        # encoder_output, hidden, syntax_hidden = model.step_encoding(src, tags)   # get the hidden state of the encoder
        encoder_output, hidden = model.step_word_encoding(src)
        _, hidden_syntax = model.step_syntax_encoding(tags)

        output_labels = []
        input_flag = [[2] for j in range(trg.shape[0])]     # 第一位的标志位
        input_flag = torch.Tensor(input_flag).long().cuda()

        output_labels = []
        out_flag = 2    # 一开始的output flag

        # hidden_syntax = syntax_hidden
        for index in range(trg.shape[1]):
            flag4decoder = embed_flag(input_flag)
            trg_step = trg[:, index, :].view(batch_size, 1, -1)
            tags_step = tags[:, index].view(batch_size, -1)
            attn_input = torch.index_select(encoder_output, 1, torch.tensor(model.config.max_len-1-index).cuda())   # 对应位置的attention

            trg_step = torch.cat([trg_step, flag4decoder], dim=-1)
            trg_step = torch.cat([trg_step, attn_input], dim=-1)


            out, hidden_syntax, hidden = model.test_decoding(trg_step, tags_step, flag4decoder, hidden_syntax=hidden_syntax, hidden=hidden)

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

