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


# Test the Bi-LSTM
def test_BiLSTM(model_num=1):
    DATA_PATH = './data/dataset_eval'
    DICT_PATH = './checkpoint/dict_20000.pkl'
    EMBEDDING_PATH_RANDOM = './model/save_embedding_97and3.ckpt'
    GPU_NUM = 0
    MODEL_PATH = './checkpoint/BiLSTM/base_seq2seq_epoch{}.ckpt'
    BATCH_SIZE = 100

    torch.cuda.set_device(GPU_NUM)

    # vocabulary类
    vocab = pickle.load(open(DICT_PATH, 'rb'))

    testset = dataset.CompresDataset(vocab=vocab, data_path=DATA_PATH)
    testloader = DataLoader(dataset=testset,
                            collate_fn=my_fn,
                            batch_size=BATCH_SIZE,
                            pin_memory=True,
                            shuffle=False)

    config = BiLSTMConfig()
    model = BiLSTMSeq2Seq(config).cuda()

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
        # hidden = tuple([state.view(config.de_num_layers, -1, config.de_hidden_size) for state in hidden])

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

    P = correct_num / all_num
    R = recall_correct / recall_all
    F1 = 2 * P * R / (P + R)
    print('Finally', BATCH_SIZE)
    print('\tPrecision is {}'.format(P))
    print('\tRecall is {}'.format(R))
    print('\tF1 is {}'.format(F1))
    print(correct_num, all_num)
    print(recall_correct, recall_all)
    return P, R, F1


if __name__ == '__main__':
    # test_BiLSTM(4)
    encoder = nn.TransformerEncoderLayer(d_model=100, nhead=4)
    # print(encoder)
    src = torch.randn(10, 1, 100)
    out = encoder(src)
    print('encoder', out.shape)

    decoder = nn.TransformerDecoderLayer(d_model=100, nhead=4)

    de_src = torch.randn(10, 1, 100)
    memory = torch.randn(5, 1, 100)
    out = decoder(de_src, memory)

    print('decoder', out.shape)
    
    tfs = nn.Transformer(d_model=100, 
                         nhead=4, 
                         num_encoder_layers=4, 
                         num_decoder_layers=4,
                         dim_feedforward=400)

    out = tfs(src, de_src)
    print('decoder', out.shape)



