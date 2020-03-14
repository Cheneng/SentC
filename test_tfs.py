
from helper import *
from model import BasicTransformer, PositionalEncoding
import torch
import torch.nn as nn
import dataset
import os
from tqdm import tqdm


def test_transformer(model, dataloader, embed, embed_labels, save_path):

    model.eval()

    C_rate_remain= 0
    C_rate_all = 0
    correct_num = 0
    batch_num = 0
    recall_correct = 0
    recall_all = 0

    add_position = PositionalEncoding(d_model=97)
    if torch.cuda.is_available():
        add_position.cuda()

    for step, (src, trg, labels) in enumerate(dataloader):
        
        flag4encoder = torch.zeros(src.shape[0], src.shape[1], 3)

        if torch.cuda.is_available():
            flag4encoder = flag4encoder.cuda()
            src = src.cuda()
            trg = trg.cuda()
            labels = labels.cuda()
        
        src = embed(src)
        trg = embed(trg)

        src = add_position(src)
        trg = add_position(trg)

        src = torch.cat([src, flag4encoder], dim=2)

        src = torch.transpose(src, 0, 1)
        trg = torch.transpose(trg, 0, 1)   
        labels = torch.transpose(labels, 0, 1)     

        trg_flag = torch.tensor([[0], [2]]).long()

        trg_flag = trg_flag.expand(2, dataloader.batch_size)

        embed_flag = embed_labels(trg_flag)

        memory = model.encode(src)

        for index in tqdm(range(1, trg.size(0))):
            index_trg = trg[:index+1]
            index_trg = torch.cat([index_trg, embed_flag], dim=-1)
            out = model.decode_last(index_trg, memory)
            last_labels = torch.max(out, -1)[1].unsqueeze(0)
            trg_flag = torch.cat([trg_flag, last_labels], 0)
            if torch.cuda.is_available():
                trg_flag = trg_flag.cuda()
            embed_flag = embed_labels(trg_flag)

            # print(trg_flag)

            # print(last_labels)
            # print(trg_flag.size())
            # print(labels.size())

        mask_matrix = (labels < 2)
        ground_truth = torch.masked_select(labels, mask_matrix)
        predict_labels = torch.masked_select(torch.max(out, 1)[1],
                                             mask_matrix)
        print(ground_truth, predict_labels)
        # raise ValueError
        C_rate_all += len(predict_labels)   # length of all sentence
        C_rate_remain += torch.sum(predict_labels).item()

        correct_num += torch.sum(predict_labels == ground_truth).item()
        batch_num += len(ground_truth)

        recall_correct += torch.sum(ground_truth & predict_labels).item()
        recall_all += torch.sum(ground_truth).item()

        P = correct_num / batch_num
        R = recall_correct / recall_all
        F1 = 2 * P * R /  (P + R)

        print('Precision {}; Recall {}; F1 {}'.format(P, R, F1))

        print('finish the step {}'.format(step))
        
    P = correct_num / batch_num
    R = recall_correct / recall_all
    F1 = 2 * P * R /  (P + R)
    C_rate = C_rate_remain / C_rate_remain

    print('Precision {}; Recall {}; F1 {}; C_rate {}'.format(P, R, F1, C_rate))

    model.train()

    with open(save_path, 'w') as f:
        f.write('Precision {}, Recall {}, F1 {}, C_rate {}'.fromat(P, R, F1, C_rate))


if __name__ == '__main__':

    # Test
    DICT_PATH = './checkpoint/dict_20000.pkl'
    TEST_DIR = './data/dataset_eval'
    EMBED_PATH = './model/save_embedding_97and3.ckpt'
    SAVE_PATH = './test_out'
    SAVE_FILE = 'demo.txt'
    SAVE_DIR = os.path.join(SAVE_PATH, SAVE_FILE)

    MODEL_PATH = './checkpoint/normal/transformers_epoch98.ckpt'

    if os.path.exists(SAVE_PATH) is False:
        os.makedirs(SAVE_PATH)

    vocab = pickle.load(open(DICT_PATH, 'rb'))
    
    data = dataset.CompresDataset(vocab=vocab, data_path=TEST_DIR, reverse_src=False)
    testloader = DataLoader(dataset=data,
                            collate_fn=my_fn,
                            batch_size=2,
                            pin_memory=True if torch.cuda.is_available() else False,
                            shuffle=True)

    model = BasicTransformer(
        d_model=100,
        nhead=10,
        num_encoder_layer=4,
        num_decoder_layer=1,
        dim_feedforward=100
    )

    model.load(MODEL_PATH)

    # word embedding
    embed = nn.Embedding(num_embeddings=20000, embedding_dim=97)
    embed.load_state_dict(torch.load(EMBED_PATH))

    embed_labels = get_flag_embed()

    if torch.cuda.is_available():
        embed = embed.cuda()
        embed_labels = embed_labels.cuda()

    test_transformer(model=model, dataloader=testloader, embed=embed, embed_labels=embed_labels,
                     save_path=SAVE_DIR)


