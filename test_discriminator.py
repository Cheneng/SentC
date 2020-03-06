import pickle
import torch
import torch.optim as optim
import torch.nn as nn
import torch.utils.data as data
from helper import my_dis_fn
from dis_dataset import DisDataset
from config import SeDiscriminatorConfig
from model import SemanticDiscriminator


config = SeDiscriminatorConfig()
model = SemanticDiscriminator(config)

dict_path = './checkpoint/dict_20000.pkl'
# embedding_path = './model/save_embedding_97and3.ckpt'

vocab = pickle.load(open(dict_path, 'rb'))
embed = nn.Embedding(20000, 100)

data_pos = DisDataset(vocab=vocab)
data_neg = DisDataset(vocab=vocab, negative=True)

pos_train_loader = data.DataLoader(data_pos, batch_size=32,
                                   collate_fn=my_dis_fn)
neg_train_loader = data.DataLoader(data_neg, batch_size=32,
                                   collate_fn=my_dis_fn)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)



for i in range(3):
    loss_step = 0
    step = 0
    correct_num = 0
    all_num = 0
    over_all_correct = 0
    over_all_num = 0

    for (sent1, head1, label1), (sent2, head2, label2) in zip(neg_train_loader, pos_train_loader):
        sent = torch.cat([sent1, sent2], dim=0)
        head = torch.cat([head1, head2], dim=0)
        label = torch.cat([label1, label2], dim=0).view(-1)
        sent = embed(sent)
        head = embed(head)

        out = model(sent, head)
        loss = criterion(out, label)
        loss_step += loss
        step += 1
        # print(out.size())
        # print(out)
        pred_index = torch.max(out, dim=1)[1]
        correct_num += torch.sum(pred_index == label).item()
        all_num += len(label)

        if step % 20 == 0:
            print(loss_step/20, end='\t')
            print("accuracy {}".format(correct_num/all_num))

            loss_step = 0

            over_all_correct += correct_num
            over_all_num += all_num

            all_num = 0
            correct_num = 0

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print("epoch {0} over all correct rate {1}".format(i, over_all_correct / over_all_num))
