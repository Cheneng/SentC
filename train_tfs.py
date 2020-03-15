


from helper import *
from model import BasicTransformer, PositionalEncoding
import torch
import torch.nn as nn
import dataset
import os
import argparse
import time

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, help='path of training dir', 
                    default='./data/train_pairs')
parser.add_argument('--test_dir', type=str, help='path of testing dir', 
                    default=None)
parser.add_argument('--dict_path', type=str, help='path of dict path', 
                    default='./checkpoint/dict_20000.pkl')
parser.add_argument('--embedding_path_random',type=str, help='path of embedding', 
                    default='./model/save_embedding_97and3.ckpt')
parser.add_argument('--save_model_path', type=str, help='path of save data', 
                    default='./checkpoint/Transformer_lr{}_b{}_head{}_layer{}_ff{}/')
parser.add_argument('--batch_size', type=int, help='batch size of the training',
                    default=50)
parser.add_argument('--head', type=int, help='head of the model',
                    default=10)
parser.add_argument('--layers', type=int, help='layers of the encoder-decoder in transformers',
                    default=4)
parser.add_argument('--dim_ffd', type=int, help='dimension of the feedforward network',
                    default=100)
parser.add_argument('--lr', type=float, help='the learning rate in training',
                    default=3e-4)
parser.add_argument('--epoch', type=int, help='the epoch number in training',
                    default=100)
parser.add_argument('--save_text_path', type=str, help='the save text of log',
                    default='./save_text_no_mask')
parser.add_argument('--decoder_layers', type=int, help='the layers of decoder',
                    default=1)

args = parser.parse_args()
print(args)


# PATH & Config
DATA_DIR = args.data_dir
TEST_DIR = args.test_dir

DICT_PATH = args.dict_path

EMBEDDING_PATH_RANDOM = args.embedding_path_random

FLAG = 'transformer'
EPOCH = 100
LR = args.lr
BATCH_SIZE = args.batch_size
HEAD = args.head
LAYERS = args.layers
DECODER_LAYERS = args.decoder_layers
FFD = args.dim_ffd
SAVE_TEXT_PATH = args.save_text_path

PRINT_STEP = 5
# SAVE_STEP = 50
GPU_NUM = 1

SAVE_MODEL_PATH = args.save_model_path.format(LR, BATCH_SIZE, HEAD, LAYERS, FFD)

if os.path.exists(SAVE_MODEL_PATH) is False:
    os.makedirs(SAVE_MODEL_PATH)

if os.path.exists(SAVE_TEXT_PATH) is False:
    os.makedirs(SAVE_TEXT_PATH)

torch.manual_seed(2)

# word embedding
embed = nn.Embedding(num_embeddings=20000, embedding_dim=97)

print('loading the embedding')
embed.load_state_dict(torch.load(EMBEDDING_PATH_RANDOM))

embed_labels = get_flag_embed()

grads = {}


# Model
model = BasicTransformer(d_model=100,
                         nhead=HEAD,
                         num_encoder_layer=LAYERS,
                         num_decoder_layer=DECODER_LAYERS,
                         dim_feedforward=FFD)

# pretrain_model_path = './checkpoint/transformers_epoch90.ckpt'

# if os.path.exists(pretrain_model_path):
#     model.load_state_dict(torch.load(pretrain_model_path, map_location=torch.device('cpu')))
#     print('loading the pretrain model...')
#     time.sleep(3)

def save_grad(name, in_, out):
    # def hook(out):
    #     grads[name] = out
    # return hook
    # print(in_)
    # print(out)
    pass

model.register_backward_hook(save_grad)

# CUDA Config
if torch.cuda.is_available() is True:
    torch.cuda.set_device(GPU_NUM)
    embed.load_state_dict(torch.load(EMBEDDING_PATH_RANDOM))
    embed = embed.cuda()
    embed_labels = embed_labels.cuda()
    model.cuda()
    print("CUDA available")

else:
    print("CUDA unavailable")

# Training Config
criterion = nn.CrossEntropyLoss(ignore_index=2)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# Training dataset
vocab = pickle.load(open(DICT_PATH, 'rb'))
data = dataset.CompresDataset(vocab=vocab, data_path=DATA_DIR, reverse_src=False)
print("the number of the training data is: {}".format(len(data)))

trainloader = DataLoader(dataset=data,
                         collate_fn=my_fn,
                         batch_size=BATCH_SIZE,
                         pin_memory=True if torch.cuda.is_available() else False,
                         shuffle=True)

# Testing dataset
if TEST_DIR is not None:
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

save_text = open(os.path.join(SAVE_TEXT_PATH, ('bs{}_h{}_l{}_ffd{}_lr{}_save_text.txt'.format(BATCH_SIZE, HEAD, LAYERS, FFD, LR))), 'w')

save_text.write('Running with batch size {}\n'.format(BATCH_SIZE))

add_position = PositionalEncoding(d_model=97)

if torch.cuda.is_available():
    add_position.cuda()


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
            # src_padding_mask = src_padding_mask.cuda()
            # tgt_padding_mask = tgt_padding_mask.cuda()

        src = embed(src)
        trg = embed(trg)

        src = add_position(src)
        trg = add_position(trg)

        src = torch.cat([src, flag4encoder], dim=2)

        flag4decoder = torch.zeros([labels.shape[0], 1]).long()                     # flag for 1st output of decoder
        if torch.cuda.is_available():
            flag4decoder = flag4decoder.cuda()

        flag4decoder = torch.cat([flag4decoder, labels[:, :-1]], dim=1)             # ignore the last output of decoder
        flag4decoder = embed_labels(flag4decoder)

        trg = torch.cat([trg, flag4decoder], dim=2)

        # The input of the transpose should be (seq_len, batch_size, embed_dim)
        src = torch.transpose(src, 0, 1)
        trg = torch.transpose(trg, 0, 1)

        tgt_mask = model.generate_square_subsequent_mask(trg.size(0))
        # tgt_mask = None

        # out = model(src, trg, tgt_mask=tgt_mask, 
        #             src_key_padding_mask=src_padding_mask,
        #             tgt_key_padding_mask=tgt_padding_mask,
        #             memory_key_padding_mask=src_padding_mask)

        out = model(src, trg, tgt_mask=tgt_mask)

        ### modify ---------------------------
        out = torch.transpose(out, 0, 1)
        out = out.reshape(-1, 2)
        labels = labels.view(-1)

        # -------------------------------------
        ###  before -------------------------
        # out = out.view(-1, 2)
        # labels = labels.view(-1)
        ###  -------------------------------

        loss = criterion(out, labels)

        print('epoch {} , loss {}'.format(epoch, loss.item()), '{0} / {1}'.format(index, len(data) // BATCH_SIZE))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # print(grads['encoder'])

        mask_matrix = (labels < 2)
        ground_truth = torch.masked_select(labels, mask_matrix)
        predict_labels = torch.masked_select(torch.max(out, 1)[1],
                                             mask_matrix)

        # print('g', ground_truth[:30])
        # print('p', predict_labels[:30])

        C_rate_all += len(predict_labels)
        C_rate_remain += torch.sum(predict_labels).item()

        correct_num += torch.sum(predict_labels == ground_truth).item()
        batch_num += len(ground_truth)

        recall_correct += torch.sum(ground_truth & predict_labels).item()
        recall_all += torch.sum(ground_truth).item()

        loss_sum += loss.item()

    model.save(SAVE_MODEL_PATH + 'transformers_epoch{}.ckpt'.format(epoch))
    print('model saved...')

    P = correct_num / batch_num
    R = recall_correct / recall_all
    F1 = 2 * P * R /  (P + R)

    save_text.write('epoch {0} loss {3}: precision {1}, recall {2}, f1 {4} \n'.format(epoch, P, R, loss_sum / len(data), F1))


save_text.close()


