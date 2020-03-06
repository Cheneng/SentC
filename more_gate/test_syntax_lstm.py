# -*- coding: utf-8 -*-

from torch.utils.data import DataLoader
from model import SyntaxLSTM
from helper import *
from dataset import SyntaxDataset
from tqdm import tqdm
import dataset
import pickle
import time
import visdom
import os


def test_syntax_lstm():
    DATA_PATH = '../data/dataset_eval'
    DICT_PATH = '../checkpoint/dict_20000.pkl'
    EMBEDDING_PATH_RANDOM = '../model/save_embedding_97and3.ckpt'
    GPU_NUM = 0
    model_epoch = 20
    MODEL_PATH = f"./checkpoint/syntax_gate_lstm/model-{model_epoch}.ckpt"

    torch.cuda.set_device(GPU_NUM)

    embed = get_word_embed().cuda()
    embed_flag = get_flag_embed().cuda()
    vocab = get_vocab()

    model = SyntaxLSTM(100, 100, 10)
    # model.load(MODEL_PATH)
    model.cuda()

    testset = SyntaxDataset(vocab=vocab, data_path='../data/dataset_eval')
    testloader = DataLoader(dataset=testset,
                             batch_size=200,
                             collate_fn=syntax_fn,
                             num_workers=2,
                             pin_memory=True,
                             shuffle=False)

    correct_num = 0
    all_num = 0
    recall_correct = 0
    recall_all = 0

    model.eval()
    for index, (src, trg, labels, tags) in enumerate(tqdm(testloader)):
        print(index)
        batch_size = src.shape[0]
        src = embed(src.cuda())
        trg = embed(trg.cuda())
        labels = labels.cuda()
        tags = tags.cuda()

        # finally get the source
        flag4encoder = torch.zeros(src.shape[0], src.shape[1], 3).cuda()
        src = torch.cat([src, flag4encoder], dim=2)

        output_labels = model.testing(src, trg, tags, embed_flag)

        # print(output_labels)

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

    print('Finally')
    print('\tPrecision is {}'.format(P))
    print('\tRecall is {}'.format(R))
    print('\tF1 is {}'.format(F1))
    print(correct_num, recall_correct)
    return P, R, F1


if __name__ == '__main__':
    test_syntax_lstm()

