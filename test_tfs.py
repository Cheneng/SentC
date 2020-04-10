
from helper import *
from model import BasicTransformer, PositionalEncoding
import torch
import torch.nn as nn
import dataset
import os
from tqdm import tqdm


def test_transformer(model, dataloader, embed, embed_labels, save_path, all_step):

    model.eval()
    cal_score = Cal_Score()

    with torch.no_grad():

        add_position = PositionalEncoding(d_model=97)
        if torch.cuda.is_available():
            add_position.cuda()
            embed = embed.cuda()
            embed_labels = embed_labels.cuda()
            model.cuda()

        for step, (src, trg, labels) in enumerate(dataloader):
            
            flag4encoder = torch.zeros(src.shape[0], src.shape[1], 3)

            trg_flag = torch.tensor([[0], [2]]).long()

            trg_flag = trg_flag.expand(2, trg.size(0))

            if torch.cuda.is_available():
                
                flag4encoder = flag4encoder.cuda()
                src = src.cuda()
                trg = trg.cuda()
                labels = labels.cuda()
                trg_flag = trg_flag.cuda()
            
            src = embed(src)
            trg = embed(trg)

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

            cal_score.update(trg_flag, labels)
            cal_score.print()
            print('finish the step {} / {}'.format(step, all_step))
            
        cal_score.print()

    with open(save_path, 'w') as f:
        f.write('Precision {}, Recall {}, F1 {}, C_rate {}'.format(P, R, F1, C_rate))


if __name__ == '__main__':

    # Test
    DICT_PATH = './checkpoint/dict_20000.pkl'
    TEST_DIR = './data/dataset_eval' if torch.cuda.is_available() else './data/train_pairs'
    EMBED_PATH = './model/save_embedding_97and3.ckpt'
    SAVE_PATH = './test_out'
    SAVE_FILE = 'demo.txt'
    SAVE_DIR = os.path.join(SAVE_PATH, SAVE_FILE)

    MODEL_PATH = './checkpoint/normal/transformers_epoch90.ckpt'
    # MODEL_PATH = './checkpoint/transformers_epoch90.ckpt'

    if os.path.exists(SAVE_PATH) is False:
        os.makedirs(SAVE_PATH)

    vocab = pickle.load(open(DICT_PATH, 'rb'))
    
    data = dataset.CompresDataset(vocab=vocab, data_path=TEST_DIR, reverse_src=False)
    testloader = DataLoader(dataset=data,
                            collate_fn=my_fn,
                            batch_size=1000 if torch.cuda.is_available() else 2,
                            # batch_size=2,
                            pin_memory=True if torch.cuda.is_available() else False,
                            shuffle=True)

    model = BasicTransformer(
        d_model=100,
        nhead=10,
        num_encoder_layer=4,
        num_decoder_layer=1,
        dim_feedforward=100
    )

    # model.load(MODEL_PATH)
    if torch.cuda.is_available():
        model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))

    # word embedding
    embed = nn.Embedding(num_embeddings=20000, embedding_dim=97)
    embed.load_state_dict(torch.load(EMBED_PATH))

    embed_labels = get_flag_embed()

    test_transformer(model=model, dataloader=testloader, embed=embed, embed_labels=embed_labels,
                     save_path=SAVE_DIR, all_step=(len(data) // testloader.batch_size))


