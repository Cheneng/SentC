# -*- coding: utf-8 -*-

import torch.optim as optim
from torch.utils.data import DataLoader
from config import *
from model import *
from helper import get_word_embed, get_flag_embed, get_vocab, get_gen_trainloader
from train_helper import Visdom_line, train_generator, train_discriminator, train_ad_generator
import dataset
import pickle
import time
import visdom
import os

def rl_train_3layer():
    vis = visdom.Visdom(env="RL_compression")
    vis_g = Visdom_line(vis=vis, win="generator", name="Generator Loss")
    vis_d = Visdom_line(vis=vis, win="discriminator", name="Discriminator Loss")
    cuda_num = 1
    get_pretrain_generator = True        # True to load the pre train model
    get_pretrain_discriminator = False
    g_save_dir = './checkpoint/GAN_train/generator/'
    d_save_dir = './checkpoint/GAN_train/discriminator/'

    # MODEL_PATH = './checkpoint/LSTM3Layers/model2.ckpt'
    BATCH_SIZE = 100
    torch.cuda.set_device(cuda_num)

    # Load the model
    g_config = LSTM3LayersConfig()
    generator = LSTM3Layers(g_config).cuda()
    # generator.load(MODEL_PATH)

    # 读取embedding
    embed = get_word_embed().cuda()
    embed_flag = get_flag_embed().cuda()   # 后三位标志位embedding
    vocab = get_vocab()

    g_criterion = nn.CrossEntropyLoss(ignore_index=2)      # ignore the index of padding
    g_optimizer = optim.Adam(generator.parameters(), g_config.lr, weight_decay=0.1)

    g_trainloader = get_gen_trainloader(vocab=vocab, batch_size=g_config.batch_size, reload=True)
    d_config = SeDiscriminatorConfig()
    discriminator = SemanticDiscriminator(d_config)
    d_trainloader = get_gen_trainloader(vocab=vocab, batch_size=d_config.batch_size, reload=False)
    if torch.cuda.is_available():
        discriminator.cuda()
    d_criterion = nn.CrossEntropyLoss(ignore_index=2)       # ignore the index of padding
    d_optimizer = optim.Adam(discriminator.parameters(), lr=d_config.lr)

    if get_pretrain_generator is False:
        print("Pre train the generator...")
        vis.text('Pre train the generator at {}'.format(time.strftime('%x %X')), win='log')
        for i in range(g_config.epoch):
            train_generator(model=generator, embed=embed, flag_embed=embed_flag, dataloader=g_trainloader, vis=vis_g,
                            criterion=g_criterion, optimizer=g_optimizer, cuda=True, name="Trainig Generator Epoch {0}".format(i))
            generator.save(g_save_dir + 'pretrain_epoch{}'.format(i))

    if get_pretrain_discriminator is False:
        if get_pretrain_generator is True:
            generator.load(g_save_dir + 'pretrain_epoch{}'.format(0))
        print("Pre train the Discriminator...")
        vis.text('Pre train the Discriminator at {}'.format(time.strftime('%x %X')), win='log')

        for j in range(d_config.epoch):
            train_discriminator(gen=generator, dis=discriminator, embed=embed, flag_embed=embed_flag,
                                dataloader=d_trainloader, vis=vis_d, criterion=d_criterion, optimizer=d_optimizer,
                                cuda=True, name="Training Discriminator Epoch {0}".format(j))
            discriminator.save(d_save_dir + 'pretrain_epoch{}'.format(j))
    else:
        print("Loading the pre train model")
        vis.text("Get the pre train model at {}".format(time.strftime('%x %X')), win='log')
        generator.load(g_save_dir + 'pretrain_epoch{}'.format(0))
        discriminator.load(d_save_dir + 'pretrain_epoch{}'.format(0))

    # TODO: Adversarial Training
    ad_batch_size = 250
    ad_train_loader = get_gen_trainloader(vocab=vocab, batch_size=ad_batch_size, reload=False)
    ad_criterion = nn.NLLLoss(ignore_index=2)
    ad_optimizer = optim.Adam(generator.parameters(), lr=0.001)

    for ad_epoch in range(5):
        for g_step in range(1):
            train_ad_generator(g_model=generator, d_model=discriminator, embed=embed,
                               flag_embed=embed_flag, dataloader=ad_train_loader, vis=vis_g, name="Ad traing Epoch {}".format(g_step),
                               test_criterion=g_criterion, criterion=ad_criterion, optimizer=g_optimizer, cuda=True)
        for d_step in range(2):
            train_discriminator(dis=discriminator, gen=generator, embed=embed, flag_embed=embed_flag,
                                dataloader=d_trainloader, vis=vis_d, criterion=d_criterion, optimizer=d_optimizer)



if __name__ == "__main__":
    rl_train_3layer()


