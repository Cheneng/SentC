import os
import torch
import torch.nn as nn
import pickle
import visdom
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import DataLoader
from model import LSTMGraphAttn
from config import GraphAttenConfig
from helper import get_word_embed, get_flag_embed, get_vocab, get_gen_trainloader, graph_fn
from train_helper import Visdom_line, train_generator, train_discriminator, train_ad_generator
from graph_dataset import GraphDataset
from tensorboardX import SummaryWriter


def train(reload_dataset=False, pretrain_model_path=None, optim_fu='adam'):
    write = SummaryWriter()

    vis = visdom.Visdom(env="Graph_Attention_compression")
    viz = Visdom_line(vis=vis, win="Graph_Attention")

    # 一些配置
    DATA_DIR = './data/train_pairs'
    DICT_PATH = './checkpoint/dict_20000.pkl'
    EMBEDDING_PATH_RANDOM = './model/save_embedding_97and3.ckpt'
    SAVE_EMBEDDING = False
    RELOAD_DATASET = reload_dataset

    SAVE_DATASET_OBJ = './data/dataset.pkl'
    SAVE_MODEL_PATH = './checkpoint/Graph_Attn/'

    PRINT_STEP = 10
    SAVE_STEP = 1
    GPU_NUM = 0

    torch.manual_seed(2)
    torch.cuda.set_device(GPU_NUM)

    config = GraphAttenConfig()

    model = LSTMGraphAttn(config)
    model.cuda()

    if os.path.exists(SAVE_MODEL_PATH) is False:
        os.makedirs(SAVE_MODEL_PATH)

    # 读取embedding
    embed = get_word_embed().cuda()
    embed_flag = get_flag_embed().cuda()
    vocab = get_vocab()

    criterion = nn.CrossEntropyLoss(ignore_index=2)
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    trainset = GraphDataset(vocab=vocab)
    trainloader = DataLoader(dataset=trainset,
                             batch_size=config.batch_size,
                             collate_fn=graph_fn,
                             pin_memory=True,
                             shuffle=True)

    global_step = 0
    for epoch in range(config.epoch):
        epoch_loss = 0
        for index, (src, trg, neighbor, labels) in enumerate(tqdm(trainloader)):
            src = embed(src.cuda())
            trg = embed(trg.cuda())
            neighbor = embed(neighbor.cuda())

            flag4encoder = torch.zeros(src.shape[0], src.shape[1], 3).cuda()
            src = torch.cat([src, flag4encoder], dim=2)

            flag4decoder = torch.zeros([labels.shape[0], 1]).long()
            flag4decoder = torch.cat([flag4decoder, labels[:, :-1]], dim=1).cuda()
            flag4decoder = embed_flag(flag4decoder)

            flag4neighbor = torch.zeros(neighbor.shape[0], neighbor.shape[1], neighbor.shape[2], 3).cuda()
            neighbor = torch.cat([neighbor, flag4neighbor], dim=-1)

            trg = torch.cat([trg, flag4decoder], dim=2)
            labels = labels.cuda()

            out = model(src, trg, neighbor)
            out = out.view(-1, 2)
            labels = labels.view(-1)
            loss = criterion(out, labels)
            epoch_loss += loss.item()
            print(loss.item())
            optimizer.zero_grad()
            loss.backward()

            optimizer.step()

            write.add_scalar('loss', loss.item(), global_step)
            global_step += 1

        model.save(SAVE_MODEL_PATH + 'model-' + str(epoch) + '.ckpt')
        write.add_scalar('epoch_loss', epoch_loss, epoch)


if __name__ == '__main__':
    train()






