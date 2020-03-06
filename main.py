# -*- coding: utf-8 -*-

import torch.optim as optim
from torch.utils.data import DataLoader
from config import Config
from model import *
from helper import *
import dataset
import pickle
import time
import visdom
import os


def train(reload_dataset=False):

    # 一些配置
    DATA_DIR = './data/train_pairs'
    DICT_PATH = './checkpoint/dict_20000.pkl'

    EMBEDDING_PATH_RANDOM = './model/save_embedding_100.ckpt'
    SAVE_EMBEDDING = True
    RELOAD_DATASET = reload_dataset

    SAVE_DATASET_OBJ = './data/dataset.pkl'
    SAVE_MODEL_PATH = './checkpoint/save_seq2seq1layer_withoutFLAG/'
    VISDOM_ENV = 'SentenceCompression-withoutFLAG'
    PRINT_STEP = 100
    SAVE_STEP = 5
    GPU_NUM = 0

    vis = visdom.Visdom(env=VISDOM_ENV)

    if os.path.exists(SAVE_MODEL_PATH) is False:
        os.makedirs(SAVE_MODEL_PATH)

    torch.manual_seed(1)
    torch.cuda.set_device(GPU_NUM)
    config = Config()

    # 模型
    model = Seq2Seq(config).cuda()

    # 预训练好的词向量读取
    embed = nn.Embedding(num_embeddings=20000, embedding_dim=100, padding_idx=0)
    if SAVE_EMBEDDING is True:
        torch.save(embed.state_dict(), EMBEDDING_PATH_RANDOM)
    else:
        embed.load_state_dict(torch.load(EMBEDDING_PATH_RANDOM))

    embed.cuda()

    criterion = nn.CrossEntropyLoss(ignore_index=2)    # ignore padding index
    optimizer = optim.Adam(model.parameters(), lr=config.lr)

    # vocabulary类
    vocab = pickle.load(open(DICT_PATH, 'rb'))

    if RELOAD_DATASET is True:
        data = dataset.CompresDataset(vocab=vocab, data_path=DATA_DIR)
        with open(SAVE_DATASET_OBJ, 'wb') as f:
            pickle.dump(data, f)
    else:
        data = pickle.load(open(SAVE_DATASET_OBJ, 'rb'))

    print('The length of the data is: {}'.format(len(data)))

    trainloader = DataLoader(dataset=data,
                             collate_fn=my_fn,
                             batch_size=config.batch_size,
                             pin_memory=True,
                             shuffle=True)

    vis.text('Running the seq2seq at {}'.format(time.strftime('%x %X')), win='log')

    episode = 0
    loss_sum = 0
    axis_index = 0
    correct_num = 0
    batch_num = 0
    recall_correct = 0
    recall_all = 0
    acc = 0     # TODO:the all correct output
    C_rate_remain = 0
    C_rate_all = 0
    save_index = 0

    for epoch in range(config.epoch):
        for src, trg, labels in trainloader:
            # print(labels)

            src = embed(src.cuda())
            trg = embed(trg.cuda())
            labels = labels.cuda()

            out, _ = model(src, trg)
            out = out.view(-1, 2)
            labels = labels.view(-1)
            loss = criterion(out, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # mask_matrix = labels.ge(0)
            mask_matrix = (labels < 2)
            ground_truth = torch.masked_select(labels, mask_matrix)
            predict_labels = torch.masked_select(torch.max(out, 1)[1],
                                                 mask_matrix)

            print('g', ground_truth)
            print('p', predict_labels)

            C_rate_all += len(predict_labels)
            C_rate_remain += torch.sum(predict_labels).item()

            correct_num += torch.sum(predict_labels == ground_truth).item()
            batch_num += len(ground_truth)
            # 训练的召回率计算
            recall_correct += torch.sum(ground_truth & predict_labels).item()
            recall_all += torch.sum(ground_truth).item()
            # 记录loss
            loss_sum += loss.item()

            if episode % PRINT_STEP == 0 and episode != 0:
                # calculate the precision P, recall R, and F1
                P = correct_num / batch_num
                R = recall_correct / recall_all
                F1 = 2 * P * R / (P + R)

                # calculate for the compression rate.
                C_rate = C_rate_remain / C_rate_all

                draw_line(vis, axis_index, (P, R, F1), names=('Precision', 'Recall', 'F1 score'))
                draw_line(vis, axis_index, (loss_sum/PRINT_STEP,), names=('Step Loss',))
                draw_line(vis, axis_index, (C_rate,), names=('Compression Rate',))

                # update
                correct_num = 0
                batch_num = 0
                recall_correct = 0
                recall_all = 0
                axis_index += 1
                loss_sum = 0
                C_rate_all = 0
                C_rate_remain = 0

                # save model
                if axis_index % SAVE_STEP == 0:
                    model.save(SAVE_MODEL_PATH + 'base_seq2seq{}.ckpt'.format(save_index))
                    save_index += 1
            episode += 1
        vis.text(time.strftime('%x %X') + 'finished the epoch {}'.format(epoch), win='log', append=True)


def test_output_file(save_dir='./output_file', save_filename='myTest.txt'):
    SAVE_PATH = save_dir
    FILE_NAME = save_filename
    DATA_PATH = './data/dataset_eval'
    DICT_PATH = './model/save_/vocab.pkl'
    MODEL_PATH = './checkpoint/save_baseline/base_seq2seq500.ckpt'
    EMBEDDING_PATH = './model/save_/embedding_8k.ckpt'
    GPU_NUM = 1

    if os.path.exists(SAVE_PATH) is False:
        os.mkdir(save_dir)

    torch.cuda.set_device(GPU_NUM)

    # vocabulary类
    vocab = pickle.load(open(DICT_PATH, 'rb'))
    config = Config()

    testset = dataset.CompresDataset(vocab=vocab, data_path=DATA_PATH)
    testloader_word = DataLoader(dataset=testset,
                                 collate_fn=my_fn,
                                 batch_size=1,
                                 pin_memory=True)

    model = Seq2Seq(config).cuda()
    model.load(MODEL_PATH)

    embed = nn.Embedding(num_embeddings=8004, embedding_dim=300)
    embed.load_state_dict(torch.load(EMBEDDING_PATH))
    embed.cuda()

    path = os.path.join(SAVE_PATH, FILE_NAME)
    save_file = open(path, 'w')

    pair_num = len(testloader_word)
    print('The number of the sentences pairs is :{}'.format(pair_num))

    for index, (src, trg, labels) in enumerate(testloader_word, 1):
        in_word = src.cuda()
        src = embed(src.cuda())
        trg = embed(trg.cuda())
        labels = labels.cuda()

        out = model(src, trg)

        out = out.view(-1, 2)
        labels = labels.view(-1)

        mask_matrix = labels.ge(0)
        ground_truth = torch.masked_select(labels, mask_matrix)
        predict_labels = torch.masked_select(torch.max(out, 1)[1],
                                             mask_matrix)

        output_list = torch.masked_select(in_word, predict_labels.byte()).tolist()
        text = torch.masked_select(in_word, ground_truth.byte()).tolist()

        sentence_list = vocab.index_list_to_sentence(in_word.squeeze().tolist())
        out_word_list = vocab.index_list_to_sentence(output_list)
        text_list = vocab.index_list_to_sentence(text)

        sentence = ' '.join(sentence_list)
        text = ' '.join(text_list)
        output_word = ' '.join(out_word_list)

        print(index, '/', pair_num)

        save_file.write('{0}\n{1}\n{2}\n{3}\n\n'.format(index, sentence, text, output_word))

    save_file.close()

if __name__ == '__main__':
    train(False)
    # test_output_file()
