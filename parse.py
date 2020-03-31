from nltk.parse.stanford import StanfordDependencyParser
from nltk.parse import stanford
import nltk
import os
from dataset import del_mac_DS
from helper import *
from nltk.tokenize import casual_tokenize
import time

#添加stanford环境变量,此处需要手动修改，jar包地址为绝对地址。
dir_path = './stanford_parser_jar'
os.environ['STANFORD_PARSER'] = os.path.join(dir_path, 'stanford-parser.jar')
os.environ['STANFORD_MODELS'] = os.path.join(dir_path, 'stanford-parser-3.5.2-models.jar')

parser = stanford.StanfordDependencyParser(model_path=os.path.join(dir_path, 'englishPCFG.ser.gz'))


def get_the_parse(sent_list_in):
    tag_dict = {}
    sentences = parser.parse_sents([sent_list_in])
    tree = [parse.tree() for parse in list(sentences)[0]]
    n_leaves = len(tree[0].leaves())
    leavepos = list(tree[0].leaf_treeposition(n) for n in range(n_leaves))

    for pos in tree[0].treepositions():
        c_word = None
        if pos not in leavepos:
            c_word = tree[0][pos].label()
        else:
            c_word = tree[0][pos]
        # print(c_word, len(pos))
        # add the word to dict
        if c_word in tag_dict.keys():
            tag_dict[c_word].insert(0, len(pos))
        else:
            tag_dict[c_word] = [len(pos)]

    return tag_dict


def parse_files(files_dir='./data/train_pairs', save_dir='./'):
    vocab = pickle.load(open('./checkpoint/dict_20000.pkl', 'rb'))
    file_names = del_mac_DS(os.listdir(files_dir))
    
    for name in file_names:
        print('loading', name)
        save_f = open(os.path.join(save_dir, name), 'w')
        with open(os.path.join(files_dir, name), 'r') as f:
            for index, line in enumerate(f, 1):
                line = line.split('\t', 1)
                sent = casual_tokenize(line[0])
                sent_dict = get_the_parse(sent)
                sent_pos = get_sent_pos(sent, sent_dict)
                print(sent, sent_pos)
                save_f.write('{0}\t{1}\t{2}\n'.format(line[0], ' '.join(map(str, sent_pos)), line[1]))

        save_f.close()

def get_sent_pos(sent, pos_dict):
    sent_pos = []
    for word in sent:
        if word not in pos_dict.keys():     # 标点等
            sent_pos.append(16)
        else:
            if len(pos_dict[word]) > 1:
                sent_pos.append(pos_dict[word].pop())
            else:
                sent_pos.append(pos_dict[word][0])
    return sent_pos


if __name__ == '__main__':
    x = "chen cheng is a good man , he done things well, he loves me".split(' ')
    # print(x)
    # tag = get_the_parse(x)
    # print(tag)
    # print(tag['he'])
    # print(tag['he'].pop())
    # print(tag['he'])
    save_path = "./data/parse_dir"
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)
    parse_files(save_dir=save_path)

