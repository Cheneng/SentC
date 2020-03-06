


from helper import *
from model import BasicTransformer
import torch
import torch.nn as nn
import dataset
import os


# PATH & Config
DATA_DIR = './data/train_pairs'
TEST_DIR = './data/train_pairs' 

DICT_PATH = './checkpoint/dict_20000.pkl'
EMBEDING_PATH_RANDOM = './model/save_embedding_97and3.ckpt'

SAVE_MODEL_PATH = './checkpoint/Transformer/'
if os.path.exists(SAVE_MODEL_PATH) is False:
    os.makedirs(SAVE_MODEL_PATH)

FLAG = 'transformer'
EPOCH = 100
BATCH_SIZE = 1
PRINT_STEP = 5
SAVE_STEP = 50
GPU_NUM = 0

torch.manual_seed(2)

# word embedding
embed = nn.Embedding(num_embeddings=20000, embedding_dim=97)
embed_labels = get_flag_embed()

# Model
model = BasicTransformer(d_model=100,
                         nhead=4,
                         num_encoder_layer=4,
                         num_decoder_layer=4,
                         dim_feedforward=400)

# CUDA Config
if torch.cuda.is_available is True:
    embed.load_state_dict(torch.load(EMBEDDING_PATH_RANDOM))
    embed = embed.cuda()
    embed_labels = embed_labels.cuda()
    model.cuda()
    torch.cuda_set_device(GPU_NUM)
    print("CUDA available")

else:
    print("CUDA unavailable")

# Training Config
criterion = nn.CrossEntropyLoss(ignore_index=2)
optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

# Training dataset
vocab = pickle.load(open(DICT_PATH, 'rb'))
data = dataset.CompresDataset(vocab=vocab, data_path=DATA_DIR)
print("the number of the training data is: {}".format(len(data)))
trainloader = DataLoader(dataset=data,
                         collate_fn=my_fn,
                         batch_size=BATCH_SIZE,
                         pin_memory=True if torch.cuda.is_available() else False,
                         shuffle=True)

# Testing dataset
testset = dataset.CompresDataset(vocab=vocab, data_path=TEST_DIR)
testloader = DataLoader(dataset=testset,
                        collate_fn=my_fn,
                        batch_size=BATCH_SIZE,
                        pin_memory=True,
                        shuffle=False)

C_rate_remain = 0
C_rate_all = 0
correct_num = 0
batch_num = 0
loss_sum = 0
recall_correct = 0
recall_all = 0

save_text = open('./{0}_b{1}_save_text.txt'.format(FLAG, BATCH_SIZE), 'w')
save_text.write('This is the line\n')

for epoch in range(EPOCH):
    C_rate_remain = 0
    C_rate_all = 0
    correct_num = 0
    batch_num = 0
    loss_sum = 0
    recall_correct = 0
    recall_all = 0
    for index, (src, trg, labels) in enumerate(trainloader, 1):

        flag4encoder = torch.zeros(src.shape[0], src.shape[1], 3)
        
        # CUDA 
        if torch.cuda.is_available():
            flag4encoder = flag4encoder.cuda()
            src = src.cuda()
            trg = trg.cuda()
            labels = labels.cuda()

        src = embed(src)
        trg = embed(trg)

        src = torch.cat([src, flag4encoder], dim=2)

        flag4decoder = torch.zeros([labels.shape[0], 1]).long()                     # flag for 1st output of decoder
        flag4decoder = torch.cat([flag4decoder, labels[:, :-1]], dim=1)             # ignore the last output of decoder
        flag4decoder = embed_labels(flag4decoder)

        trg = torch.cat([trg, flag4decoder], dim=2)

        # The input of the transpose should be (seq_len, batch_size, embed_dim)
        src = torch.transpose(src, 0, 1)
        trg = torch.transpose(trg, 0, 1)

        tgt_mask = model.generate_square_subsequent_mask(trg.size(0))
        out = model(src, trg, tgt_mask=tgt_mask)

        out = out.view(-1, 2)
        labels = labels.view(-1)

        loss = criterion(out, labels)
        print(loss.item(), '{0} / {1}'.format(index, len(data) // BATCH_SIZE))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        mask_matrix = (labels < 2)
        ground_truth = torch.masked_select(labels, mask_matrix)
        predict_labels = torch.masked_select(torch.max(out, 1)[1],
                                             mask_matrix)

        print('g', ground_truth[:30])
        print('p', predict_labels[:30])

        C_rate_all += len(predict_labels)
        C_rate_remain += torch.sum(predict_labels).item()

        correct_num += torch.sum(predict_labels == ground_truth).item()
        batch_num += len(ground_truth)

        recall_correct += torch.sum(ground_truth & predict_labels).item()
        recall_all += torch.sum(ground_truth).item()

        loss_sum += loss.item()

        if index % SAVE_STEP == 0:
            model.save(SAVE_MODEL_PATH + 'transformers_epoch{}.ckpt'.format(epoch))

    P = correct_num / batch_num
    R = recall_correct / recall_all
    F1 = 2 * P * R /  (P + R)

    save_text.write('epoch {0} loss {3}: precision {1}, recall {2}, f1 {4} \n'.format(epoch, P, R, loss_sum / len(data), F1))

    # test epoch model
    # print('TESTING MODEL')
    # for i, (src, trg, labels) in enumerate(testloader):
    #     if torch.cuda.is_available() is True:
    #         src = 

save_text.close()

