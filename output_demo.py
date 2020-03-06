import torch
from config import *
from model import *
import pickle
from helper import *
from nltk.tokenize import casual_tokenize
import copy

test_data = ['A woman was injured by a falling tree in the Gresham neighborhood, according to the Chicago Fire Department.',

             "A new wave of attacks across Iraq killed at least 20 people and wounded dozens on Monday as the government \
             pressed on with its offensive to hunt down al-Qaida-linked militants in the country's volatile western desert.",

             "Vine, the mobile app owned by Twitter, has banned sexually explicit content, effective immediately.",

             "Exercise may be just as good as medication to treat heart disease and should be included as a comparison \
             when new drugs are being developed and tested, scientists said on Wednesday.",

             "Silver Standard Resources Inc. announces today that it has entered into a Purchase and Sale Agreement \
             with subsidiaries of Goldcorp Inc. and Barrick Gold Corporation to purchase 100% of the Marigold mine, \
             a producing gold mine in Nevada, USA for cash consideration of $275 million.",

             "BMW Group Middle East has confirmed that the new BMW 5 Series will go on sale in the Middle East in September.",

             "Winger Phil Kessel was absent from practice at the Mastercard Center in Etobicoke, ON Monday morning, but it was not because of an injury."]

ground_truth = ['A woman was injured by a falling tree.',
                'A wave of attacks across Iraq killed at least 20 people.',
                "Vine has banned sexually explicit content.",
                "Exercise may be good as medication to treat heart disease.",
                "Silver Standard Resources Inc. has entered to purchase 100% of the Marigold mine.",
                "The new BMW 5 Series will go on sale in September.",
                "Phil Kessel was absent from practice."]


model_path = {1: './checkpoint/LSTM3Layers/model4.ckpt',
              2: './checkpoint/BiLSTM/base_seq2seq_epoch4.ckpt',
              3: './checkpoint/BiLSTMAttn/base_seq2seq_epoch25.ckpt',
              4: './checkpoint/BiLSTM_T_Attn/base_seq2seq_epoch3.ckpt'}

DICT_PATH = './checkpoint/dict_20000.pkl'
vocab = pickle.load(open(DICT_PATH, 'rb'))

EMBEDDING_PATH_RANDOM = './model/save_embedding_97and3.ckpt'
embed = nn.Embedding(num_embeddings=20000, embedding_dim=97)
embed.load_state_dict(torch.load(EMBEDDING_PATH_RANDOM))
embed.cuda()
embed_flag = get_flag_embed()
embed_flag.cuda()

config1 = LSTM3LayersConfig()
config2 = BiLSTMConfig()
config3 = BiLSTMAttnConfig()
config4 = BiLSTMTAttnConfig()

model1 = LSTM3Layers(config1).cuda()
model2 = BiLSTMSeq2Seq(config2).cuda()
model3 = BiLSTMAttn(config3).cuda()
model4 = BiLSTMTAttn(config4).cuda()

model1.load(model_path[1])
model2.load(model_path[2])
model3.load(model_path[3])
model4.load(model_path[4])


for jj, data in enumerate(test_data):
    src_data = vocab.sent_to_index(data)

    re_data = copy.deepcopy(src_data)

    trg = vocab.sent_to_index(data)
    trg.insert(0, 1)

    flag_data = copy.deepcopy(trg)

    src = padding_sequence([src_data], max_len=210)
    src = torch.Tensor(src).long().cuda()
    src = embed(src)

    re_data.reverse()
    src_data = padding_sequence([re_data], max_len=210)
    src_reverse = torch.Tensor(src_data).long().cuda()
    src_reverse = embed(src_reverse)

    flag4src = torch.zeros(src.shape[0], src.shape[1], 3).cuda()
    src = torch.cat([src, flag4src], dim=2)
    src_reverse = torch.cat([src_reverse, flag4src], dim=2)

    output1, hidden1 = model1.step_encoding(src)
    output2, hidden2 = model2.step_encoding(src)
    output3, hidden3 = model3.step_encoding(src)
    output4, hidden4 = model4.step_encoding(src_reverse)

# --------------------------------------------------------
    input_flag = [[2]]
    input_flag = torch.Tensor(input_flag).long().cuda()
    model1_output_labels = []
    for index in trg:
        # Prepare for the input
        flag4encoder = embed_flag(input_flag)
        # select_elem = torch.index_select(trg, 1, torch.tensor(index).cuda())
        select_elem = torch.Tensor([[index]]).long().cuda()
        decoder_input = embed(select_elem)
        decoder_input = torch.cat([decoder_input, flag4encoder], dim=2)

        out, hidden = model1.step_decoding(decoder_input, hidden1)
        input_flag = torch.max(out, 2)[1]
        model1_output_labels.append(input_flag.item())
# -----------------------------------------------------
    input_flag = [[2]]
    input_flag = torch.Tensor(input_flag).long().cuda()
    model2_output_labels = []
    for index in trg:
        # Prepare for the input
        flag4encoder = embed_flag(input_flag)
        select_elem = torch.Tensor([[index]]).long().cuda()
        decoder_input = embed(select_elem)
        decoder_input = torch.cat([decoder_input, flag4encoder], dim=2)

        out, hidden2 = model2.step_decoding(decoder_input, hidden2)
        input_flag = torch.max(out, 2)[1]
        model2_output_labels.append(input_flag.item())

# -------------------------------------------------------
    input_flag = [[2]]
    input_flag = torch.Tensor(input_flag).long().cuda()
    model3_output_labels = []
    for index in trg:
        # Prepare for the input
        flag4encoder = embed_flag(input_flag)
        select_elem = torch.Tensor([[index]]).long().cuda()
        decoder_input = embed(select_elem)
        decoder_input = torch.cat([decoder_input, flag4encoder], dim=2)

        out, hidden3 = model3.attn_step_decoding(decoder_input, output3, hidden3)
        input_flag = torch.max(out, 2)[1]
        model3_output_labels.append(input_flag.item())

# -------------------------------------------------------
    input_flag = [[2]]
    input_flag = torch.Tensor(input_flag).long().cuda()
    model4_output_labels = []
    for i, index in enumerate(trg):
        # Prepare for the input
        flag4encoder = embed_flag(input_flag)
        select_elem = torch.Tensor([[index]]).long().cuda()
        decoder_input = embed(select_elem)
        decoder_input = torch.cat([decoder_input, flag4encoder], dim=2)

        out, hidden4 = model4.attn_step_decoding(i, decoder_input, output4, hidden4)
        input_flag = torch.max(out, 2)[1]
        model4_output_labels.append(input_flag.item())

# -------------------------------------------------------

    # print(model1_output_labels)
    # print(model2_output_labels)
    # print(model3_output_labels)
    # print(model4_output_labels)

    # print(flag_data)
    #
    # data_split = casual_tokenize(data)
    # out_1 = []
    # for data, pred_labels, oov_flag in zip(data_split, model1_output_labels[1:], flag_data):
    #     if pred_labels == 1:
    #         # out_1.append(data if oov_flag != 3 else data+'(OOV)')
    #         out_1.append(data)
    # print(' '.join(out_1))

    def get_compression_data(origin_data, output_labels, flag_data, name=''):
        data_split = casual_tokenize(origin_data)
        out_1 = []
        for data, pred_labels, oov_flag in zip(data_split, output_labels[1:], flag_data):
            if pred_labels == 1:
                # out_1.append(data if oov_flag != 3 else data+'(OOV)')
                out_1.append(data)
        print(name, end='\t\t\t')
        print(' '.join(out_1))

    print('origin: ', data)
    print('ground truth: ', ground_truth[jj])
    get_compression_data(data, model1_output_labels, flag_data, '3-LSTM')
    get_compression_data(data, model2_output_labels, flag_data, 'Bi-LSTM')
    get_compression_data(data, model3_output_labels, flag_data, 'Attention')
    get_compression_data(data, model4_output_labels, flag_data, 't-Attention')
    print()
