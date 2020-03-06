import os
import torch
import torch.nn as nn
import pickle
import visdom
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import DataLoader
from model import BiLSTMTSyntaxAttn
from config import BiLSTMTAttnSyntaxConfig
from helper import get_word_embed, get_flag_embed, get_vocab, get_gen_trainloader, syntax_fn
from train_helper import Visdom_line, train_generator, train_discriminator, train_ad_generator
from dataset import SyntaxDataset
from tensorboardX import SummaryWriter


def train(reload_dataset=False, pretrain_model_path=None, optim_fu='adam'):
    write = SummaryWriter(comment='t_attn')

    # vis = visdom.Visdom(env="syntax_compression")
    # viz = Visdom_line(vis=vis, win="syntax_Attention")

    # 一些配置
    DATA_DIR = './data/train_pairs'
    DICT_PATH = './checkpoint/dict_20000.pkl'
    EMBEDDING_PATH_RANDOM = './model/save_embedding_97and3.ckpt'

    SAVE_EMBEDDING = False
    RELOAD_DATASET = reload_dataset

    SAVE_DATASET_OBJ = './data/dataset.pkl'
    SAVE_MODEL_PATH = './checkpoint/syntax_t_Attn/'

    PRINT_STEP = 10
    SAVE_STEP = 1
    GPU_NUM = 0

    torch.manual_seed(2)
    torch.cuda.set_device(GPU_NUM)

    config = BiLSTMTAttnSyntaxConfig()
    model = BiLSTMTSyntaxAttn(config)
    model.cuda()

    if os.path.exists(SAVE_MODEL_PATH) is False:
        os.makedirs(SAVE_MODEL_PATH)

    # 读取embedding
    embed = get_word_embed().cuda()
    embed_flag = get_flag_embed().cuda()

    vocab = get_vocab()

    criterion = nn.CrossEntropyLoss(ignore_index=2)

    if pretrain_model_path is not None:
        print('Loading the pre train model', pretrain_model_path)
        model.load(pretrain_model_path)
        model.embed.weight.requires_grad = True
        parameters = model.parameters()
        optimizer = optim.SGD(parameters, lr=0.000001)
    else:
        parameters = filter(lambda p: p.requires_grad, model.parameters())
        optimizer = optim.Adam(parameters, lr=1e-3)


    trainset = SyntaxDataset(vocab=vocab)
    trainloader = DataLoader(dataset=trainset,
                             batch_size=config.batch_size,
                             collate_fn=syntax_fn,
                             pin_memory=True,
                             num_workers=3,
                             shuffle=True)

    global_step = 0
    for epoch in range(config.epoch):
        epoch_loss = 0
        for index, (src, trg, labels, tags) in enumerate(tqdm(trainloader)):
            src = embed(src.cuda())
            trg = embed(trg.cuda())
            tags = tags.cuda()

            flag4encoder = torch.zeros(src.shape[0], src.shape[1], 3).cuda()
            src = torch.cat([src, flag4encoder], dim=2)

            flag4decoder = torch.zeros([labels.shape[0], 1]).long()     # decoder最前面插入一个起始全0
            flag4decoder = torch.cat([flag4decoder, labels[:, :-1]], dim=1).cuda()
            flag4decoder = embed_flag(flag4decoder)

            trg = torch.cat([trg, flag4decoder], dim=2)     # 插入最后三位标志位
            labels = labels.cuda()

            out = model(src, trg, tags, flag4decoder)
            out = out.view(-1, 2)
            labels = labels.view(-1)
            loss = criterion(out, labels)
            epoch_loss += loss.item()
            print(loss.item())
            optimizer.zero_grad()
            loss.backward()

            optimizer.step()

            write.add_scalar('loss_t_syntax', loss.item(), global_step)
            global_step += 1

        model.save(SAVE_MODEL_PATH + 'model-' + str(epoch + model_epoch) + '.ckpt')
        write.add_scalar('epoch_loss', epoch_loss, epoch)


if __name__ == '__main__':
    model_epoch = 0
    model_path = f"./checkpoint/syntax_Attn/model-{model_epoch}.ckpt"
    # train(pretrain_model_path=model_path)
    train()

