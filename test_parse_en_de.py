
from helper import *
from model import BasicTransformer, PositionalEncoding
import torch
import torch.nn as nn
import dataset
import os
from tqdm import tqdm


def test_transformer(model, dataloader, embed, embed_labels, embed_parse, save_path, all_step):

    model.eval()
    C_rate_remain= 0
    C_rate_all = 0
    correct_num = 0
    batch_num = 0
    recall_correct = 0
    recall_all = 0
    PP = 0
    PP_all = 0

    with torch.no_grad():

        add_position = PositionalEncoding(d_model=97)
        if torch.cuda.is_available():
            add_position.cuda()
            embed = embed.cuda()
            embed_parse = embed_parse.cuda()
            embed_labels = embed_labels.cuda()
            model.cuda()

        for step, (src, parse, trg, labels) in enumerate(dataloader):
            
            flag4encoder = torch.zeros(src.shape[0], src.shape[1], 3)

            trg_flag = torch.tensor([[0], [2]]).long()

            trg_flag = trg_flag.expand(2, trg.size(0))

            if torch.cuda.is_available():
                
                flag4encoder = flag4encoder.cuda()
                src = src.cuda()
                trg = trg.cuda()
                parse = parse.cuda()
                labels = labels.cuda()
                trg_flag = trg_flag.cuda()
            
            src = embed(src)
            trg = embed(trg)

            # parse
            parse_in = embed_parse(parse)

            flag_de = torch.zeros([labels.shape[0], 1]).long().cuda()
            flag_de = torch.cat([flag_de, parse[:, :-1]], dim=1)
            parse_de = embed_parse(flag_de)

            src += parse_in
            trg += parse_de

            src = add_position(src)
            trg = add_position(trg)

            src = torch.cat([src, flag4encoder], dim=2)

            src = torch.transpose(src, 0, 1)
            trg = torch.transpose(trg, 0, 1)   
            labels = torch.transpose(labels, 0, 1)     

            embed_flag = embed_labels(trg_flag)

            memory = model.encode(src)

            for index in tqdm(range(1, trg.size(0))):
                index_trg = trg[:index+1]
                index_trg = torch.cat([index_trg, embed_flag], dim=-1)

                tgt_mask = model.generate_square_subsequent_mask(index+1)

                out = model.decode_last(index_trg, memory, tgt_mask=tgt_mask)
                last_labels = torch.max(out, -1)[1].unsqueeze(0)
                trg_flag = torch.cat([trg_flag, last_labels], 0)
                embed_flag = embed_labels(trg_flag)

            trg_flag = trg_flag[1:, :]

            labels = labels.detach()
            trg_flag = trg_flag.detach()

            mask_matrix = (labels < 2)
            ground_truth = torch.masked_select(labels, mask_matrix)
            predict_labels = torch.masked_select(trg_flag,
                                                mask_matrix)
            print(ground_truth, predict_labels)
            C_rate_all += len(predict_labels)   # length of all sentence
            C_rate_remain += torch.sum(predict_labels).item()

            correct_num += torch.sum(predict_labels == ground_truth).item()
            batch_num += len(ground_truth)

            p_truth = torch.masked_select(ground_truth, (predict_labels == 1))
            PP += torch.sum(p_truth).float()
            PP_all += torch.sum(predict_labels)

            print(PP, PP_all)

            recall_correct += torch.sum(ground_truth & predict_labels).item()
            recall_all += torch.sum(ground_truth).item()

            P = correct_num / batch_num
            R = recall_correct / recall_all
            F1 = 2 * P * R /  (P + R)

            prec = PP / PP_all

            print('Precision {}; Recall {}; F1 {}; pp {}'.format(P, R, F1, prec))

            print('finish the step {} / {}'.format(step, all_step))
            
        P = correct_num / batch_num
        R = recall_correct / recall_all
        F1 = 2 * P * R /  (P + R)
        prec = PP / PP_all
        C_rate = C_rate_remain / C_rate_all

        print('Precision {}; Recall {}; F1 {}; C_rate {}, pp {}'.format(P, R, F1, C_rate, prec))

    model.train()

    with open(save_path, 'w') as f:
        f.write('Precision {}, Recall {}, F1 {}, C_rate {}'.format(P, R, F1, C_rate))


if __name__ == '__main__':


    # Test
    DICT_PATH = './checkpoint/dict_20000.pkl'
    TEST_DIR = './data/parse_eval' if torch.cuda.is_available() else './data/parse_dir'
    EMBED_PATH = './model/save_embedding_97and3.ckpt'
    PARSE_PATH = './model/save_parse_97.ckpt'
    SAVE_PATH = './test_out'
    SAVE_FILE = 'demo.txt'
    SAVE_DIR = os.path.join(SAVE_PATH, SAVE_FILE)

    # MODEL_PATH = './checkpoint/normal/transformers_epoch90.ckpt'
    # MODEL_PATH = './checkpoint/transformers_epoch90.ckpt'

    MODEL_PATH = './checkpoint/Parse_Transformer_lr0.0003_b200_head10_layer2_ff100/transformers_epoch90.ckpt'

    if os.path.exists(SAVE_PATH) is False:
        os.makedirs(SAVE_PATH)

    vocab = pickle.load(open(DICT_PATH, 'rb'))
    
    # data = dataset.CompresDataset(vocab=vocab, data_path=TEST_DIR, reverse_src=False)
    data = dataset.CompresParseDataset(vocab=vocab, data_path=TEST_DIR)

    testloader = DataLoader(dataset=data,
                            collate_fn=my_parse_fn,
                            batch_size=1000 if torch.cuda.is_available() else 2,
                            pin_memory=True if torch.cuda.is_available() else False,
                            shuffle=True)

    model = BasicTransformer(
        d_model=100,
        nhead=10,
        num_encoder_layer=2,
        num_decoder_layer=2,
        dim_feedforward=100
    )

    # model.load(MODEL_PATH)
    if torch.cuda.is_available():
        model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))

    # word embedding
    embed = nn.Embedding(num_embeddings=20000, embedding_dim=97)
    embed.load_state_dict(torch.load(EMBED_PATH))

    # parse embedding
    embed_parse = nn.Embedding(num_embeddings=17, embedding_dim=97)
    embed_parse.load_state_dict(torch.load(PARSE_PATH))

    embed_labels = get_flag_embed()


    test_transformer(model=model, dataloader=testloader, embed=embed, embed_labels=embed_labels, embed_parse=embed_parse,
                     save_path=SAVE_DIR, all_step=(len(data) // testloader.batch_size))


