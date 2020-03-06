import torch
import torch.nn.functional as F
import time
import numpy as np
from helper import *

def train_generator(model, embed, flag_embed, dataloader, criterion, optimizer,
                    cuda=False, vis=None, name='Train', print_step=5):
    loss_sum = 0
    for step, (src, trg, labels) in enumerate(dataloader, 1):
            if cuda is True:
                src = src.cuda()
                trg = trg.cuda()
                labels = labels.cuda()

            src = embed(src)
            trg = embed(trg)

            # Add 3-all-zero vector to the end of the input sequence.
            flag4encoder = torch.zeros(src.shape[0], src.shape[1], 3).cuda()
            src = torch.cat([src, flag4encoder], dim=2)

            # Add the former flag represent as the first input of the sequence.
            flag4decoder = torch.zeros([labels.shape[0], 1]).long().cuda()
            flag4decoder = torch.cat([flag4decoder, labels[:, :-1]], dim=1)
            flag4decoder = flag_embed(flag4decoder)
            trg = torch.cat([trg, flag4decoder], dim=2)

            out, _ = model(src, trg)
            out = out.view(-1, 2)
            labels = labels.view(-1)
            loss = criterion(out, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_sum += loss.item()

            if step % print_step == 0:
                if vis is not None:
                    vis.update(loss_sum/print_step)
                print("{0} step {1} Loss {2}".format(name, step, loss_sum/print_step))
                loss_sum = 0
    vis.log_time(name="Finish Generator Epoch")


def train_discriminator(dis, gen, embed, flag_embed, dataloader, criterion, optimizer,
                        cuda=False, vis=None, name="Train", print_step=5):
    loss_sum = 0
    for step, (src, trg, labels) in enumerate(dataloader, 1):
        if cuda is True:
            src = src.cuda()
            trg = trg.cuda()
            labels = labels.cuda()

        src = embed(src)
        trg = embed(trg)

        mask_matrix = (labels < 2)
        flag4encoder = torch.zeros(src.shape[0], src.shape[1], 3).cuda()
        src = torch.cat([src, flag4encoder], dim=2)

        _, hidden = gen.step_encoding(src)  # to get the encoder hidden state

        flag4decoder = [[2] for j in range(trg.shape[0])]
        flag4decoder = torch.Tensor(flag4decoder).long().cuda()

        output_labels = []
        output_rate = []

        for i in range(trg.shape[1]):
            flag4decoder = flag_embed(flag4decoder)
            select_elem = torch.index_select(trg, 1, torch.tensor(i).cuda())
            current_trg = torch.cat([select_elem, flag4decoder], dim=2)

            out, hidden = gen.step_decoding(current_trg, hidden)
            output_rate.append(out)
            flag4decoder = torch.max(out, 2)[1]
            output_labels.append(flag4decoder)

        # output_rate = torch.cat(output_rate, dim=1)
        # output_rate = output_rate.view(-1, output_rate.shape[-1])
        # sample_label = output_rate.multinomial(1)
        # sample_label = sample_label.view_as(labels)

        # use the max to get the real data
        sample_label = torch.cat(output_labels, dim=1)

        # Get the fake data
        # fake_data = torch.zeros_like(output_labels)
        fake_data = torch.zeros_like(sample_label)
        fake_data.masked_scatter_(mask_matrix, sample_label)
        fake_data = torch.from_numpy(np.flip(fake_data.cpu().numpy(), axis=1).copy()).long()    # fake data

        fake_labels = torch.zeros(fake_data.shape[0]).long()

        # Get the real data
        real_data = torch.zeros_like(sample_label)
        real_data.masked_scatter_(mask_matrix, labels)
        real_data = torch.from_numpy(np.flip(real_data.cpu().numpy(), axis=1).copy())
        if cuda is True:
            real_data = real_data.cuda()
            fake_data = fake_data.cuda()
        real_data = embed(real_data)
        fake_data = embed(fake_data)
        real_labels = torch.ones(real_data.shape[0]).long()

        # finally concatnate to get the input data
        origin_data = torch.cat([src, src], dim=0)
        compression_data = torch.cat([fake_data, real_data], dim=0)
        compress_labels = torch.cat([fake_labels, real_labels], dim=0)
        flag_compression = torch.zeros(compression_data.shape[0], compression_data.shape[1], 3)

        if cuda is True:
            origin_data = origin_data.cuda()
            compression_data = compression_data.cuda()
            compress_labels = compress_labels.cuda()
            flag_compression = flag_compression.cuda()

        # add all zeros the the last three dimensions
        compression_data = torch.cat([compression_data, flag_compression], dim=-1)

        # Train the discriminator
        out = dis(origin_data, compression_data)
        loss = criterion(out, compress_labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_sum += loss.item()

        if step % print_step == 0:
            if vis is not None:
                vis.update(loss_sum / print_step)
            print("{0} step {1} Loss {2}".format(name, step, loss_sum / print_step))
            loss_sum = 0
    vis.log_time(name="Finish Discriminator Epoch")

def train_ad_generator(g_model, d_model, embed, flag_embed, dataloader, test_criterion, criterion, optimizer,
                       cuda=False, vis=None, name='Adversarial Train', print_step=5):
    loss_sum = 0
    loss_sum_test = 0
    for step, (src, trg, labels) in enumerate(dataloader, 1):
        if cuda is True:
            src = src.cuda()
            trg = trg.cuda()
            labels = labels.cuda()

        src = embed(src)
        trg = embed(trg)

        #############################
        # output the testing loss
        # Add the former flag represent as the first input of the sequence.
        flag4decoder_test = torch.zeros([labels.shape[0], 1]).long().cuda()
        flag4decoder_test = torch.cat([flag4decoder_test, labels[:, :-1]], dim=1)
        flag4decoder_test = flag_embed(flag4decoder_test)
        trg_test = torch.cat([trg, flag4decoder_test], dim=2)

        flag4encoder = torch.zeros(src.shape[0], src.shape[1], 3).cuda()
        src = torch.cat([src, flag4encoder], dim=2)

        out, _ = g_model(src, trg_test)
        out = out.view(-1, 2)
        labels_test = labels.view(-1)
        loss = test_criterion(out, labels_test)
        loss_sum_test += loss.item()
        #############################

        _, hidden = g_model.step_encoding(src)  # to get the encoder hidden state

        flag4decoder = [[2] for j in range(trg.shape[0])]
        flag4decoder = torch.Tensor(flag4decoder).long().cuda()

        output_labels = []
        output_rate = []
        for i in range(trg.shape[1]):
            flag4decoder = flag_embed(flag4decoder)
            select_elem = torch.index_select(trg, 1, torch.tensor(i).cuda())
            current_trg = torch.cat([select_elem, flag4decoder], dim=2)

            out, hidden = g_model.step_decoding(current_trg, hidden)
            output_rate.append(out)
            flag4decoder = torch.max(out, 2)[1]
            # TODO: Whether to sample data rather than the max?
            # output_labels.append(flag4decoder)


        out = torch.cat(output_rate, dim=1)
        output_rate = F.softmax(out, dim=-1)
        # output_labels = torch.cat(output_labels, dim=1)
        mask_matrix = (labels < 2)
        output_rate = output_rate.view(-1, output_rate.shape[-1])

        # sample the data once a time. TODO: more sample data
        sample_label = output_rate.multinomial(1)
        sample_label = sample_label.view_as(mask_matrix)

        input_data = torch.zeros_like(sample_label)
        input_data.masked_scatter_(mask_matrix, sample_label)

        input_data = torch.from_numpy(np.flip(input_data.cpu().numpy(), axis=1).copy()).long()  # fake data
        flag_comp = torch.zeros(input_data.shape[0], input_data.shape[1], 3)

        if cuda is True:
            input_data = input_data.cuda()
            flag_comp = flag_comp.cuda()
        embed_data = embed(input_data)
        embed_data = torch.cat([embed_data, flag_comp], dim=-1)

        d_out = d_model(src, embed_data)
        reward = F.softmax(d_out, dim=-1)       # get the reward from the output
        reward = reward[:, 1].unsqueeze(-1)
        # print(reward)
        pred_out = reward.detach()    # avoid gradient
        pred_out = torch.cat([pred_out for i in range(out.shape[1])], dim=1)
        pred_out = pred_out.unsqueeze(-1)

        d_out = pred_out * F.log_softmax(out, dim=-1)

        d_out = d_out.view(-1, d_out.shape[-1])
        input_data = input_data.view(-1)
        loss = criterion(d_out, input_data)
        loss_sum += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % print_step == 0:
            print("{0} loss test {1}".format(name, loss_sum_test / print_step))
            print(loss_sum / print_step)
            loss_sum = 0
            loss_sum_test = 0



class Visdom_line(object):
    def __init__(self, vis, win, start_step=0, name="Line_1"):
        """
        :param vis: (object) visdom.Visdom object
        :param win: (str) name of the window
        :param start_step: (int) the begin of the step
        :param name: (str) the name of the line
        """
        self._vis = vis
        self._win = win
        self._start_step = start_step
        self._name = name

    def update(self, y):
        if self._start_step == 0:
            self._vis.line(X=torch.Tensor([self._start_step]),
                           Y=y if isinstance(y, torch.Tensor) else torch.Tensor([y]),
                           win=self._win,
                           name="%s" % self._name,
                           opts=dict(legend=[self._name]))
        else:
            self._vis.updateTrace(X=torch.Tensor([self._start_step]),
                                  Y=y if isinstance(y, torch.Tensor) else torch.Tensor([y]),
                                  win=self._win,
                                  name="%s" % self._name)
        self._start_step += 1

    def log_time(self, name="finish"):
        self._vis.text("{0} at {1}".format(name, time.strftime("%x %X")), win='log', append=True)


if __name__ == '__main__':
    pass