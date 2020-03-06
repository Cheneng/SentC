import torch
import os
import pickle
from helper import Vocabulary


# 获得到词向量
def get_word_embedding(embed_dim=300, embedding_path='./embedding', input_dict=None):
    # embedding dir
    EMBEDDING_DIR = embedding_path
    embedding_file_name = {50: 'glove.6B.50d.txt',
                           100: 'glove.6B.100d.txt',
                           200: 'glove.6B.200d.txt',
                           300: 'glove.6B.300d.txt'}
    try:
        file_name = embedding_file_name[embed_dim]
    except KeyError as e:
        raise KeyError(e, '\nthe input dimension must in [50, 100, 200, 300]')

    embedding_file = os.path.join(EMBEDDING_DIR, file_name)
    print('open the embedding file in', embedding_file)

    name_dict = {}

    with open(embedding_file, 'r') as f:
        for index, line in enumerate(f):
            temp = line.split(' ')
            name = temp[0]
            embedding = list(map(float, temp[1:]))
            name_dict[name] = embedding
            print(index)
    print(name_dict["'s"])

    # 按照字典的顺序组成embedding
    if input_dict is not None:
        dict_size = len(input_dict)
        zeros = torch.zeros(1, embed_dim).float()   # <PAD>为全0向量
        embed = torch.randn(3, embed_dim)     # 前4个字符为保留字符分别为<PAD>, <SOS>, <EOS>, <OOV>
        embed = torch.cat([zeros, embed], dim=0)

        count4unknow = 0
        for i in range(4, dict_size):
            try:
                embed_list = name_dict[input_dict[i].lower()]
                embed_temp = torch.FloatTensor([embed_list])
            except KeyError:
                print('The word wasn\'t in the pre_train embedding', end='\t')
                count4unknow += 1
                print(input_dict[i].lower())
                embed_temp = torch.randn([1, embed_dim])

            embed = torch.cat([embed, embed_temp], dim=0)

        if embed.size() == torch.Size([dict_size, embed_dim]):
            print('Done')
            print('The dict size is: ', dict_size)
            print('oov word is ', count4unknow)
        else:
            print('ths size of the embedding is: ', embed.size())
    return embed


def make_num_data(data_path='./data/data_obj',
                  save_path='./model/save_/'):

    DATA_PATH = data_path

    file_name = os.listdir(DATA_PATH)
    file_name = sorted(file_name)
    vocab = Vocabulary()

    # create the dict
    for name in file_name:
        print(name)
        with open(os.path.join(DATA_PATH, name), 'rb') as f:
            doc = pickle.load(f)
            for line in doc.origin:
                vocab.add_sentence(line)

    vocab.build_standard_dict(save_num=8000)
    embed = get_word_embedding(input_dict=vocab.index2word)

    with open(save_path+'vocab.pkl', 'wb') as f:
        pickle.dump(vocab, f)

    word_embed = torch.nn.Embedding.from_pretrained(embeddings=embed)
    torch.save(word_embed.state_dict(), save_path+'embedding_8k.ckpt')

if __name__ == '__main__':
    get_word_embedding()
    # make_num_data()