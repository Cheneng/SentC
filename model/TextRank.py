import nltk.tokenize as tokenize
from nltk import pos_tag
import numpy as np
import torch


class TextRank(object):
    def __init__(self, span=2, alpha=0.85, iter_num=3):
        self._word2num = {}
        self.num2word = {}
        self.word_len = 0
        self.matrix = None
        self.span = span
        self.alpha = alpha          # 转移系数
        self.iter_num = iter_num

    def get_sequential_rank(self, sentence, vocab=None, reverse=False):
        score = []
        word_list = self.tokenize_word(sentence.strip())            # split the words to list.
        word_index_list = map(lambda index: word2num_dict[index], word_list)
        word_num = len(word_list)                     # the word dict num
        # get the tags of the words
        pos_words = pos_tag(word_list, tagset='universal')  # get the tags of the words.
        remain_words = self._filter_words(pos_words)
        pairs_list = self._get_pairs(remain_words)          # get the pairs
        trans_matrix = self.get_matrix(pairs_list)          # 构造权重矩阵
        output = []

        for i in range(word_num):
            word_c = word_list[i]   # 当前选择的词
            iter_matrix = torch.ones((self.word_len, 1)) / self.word_len  # 初始转移矩阵

            if word_c in remain_words:  # 判断这个词是否被用来构图
                topic_index = self.get_word_index(word_c)
                init_matrix = torch.zeros((self.word_len, 1))
                init_matrix[topic_index][0] = 1     # 指定当前的标志位

                for _ in range(self.iter_num):  # 进行迭代
                    iter_matrix = self.iter_matrix(trans_matrix, iter_matrix, init_matrix)
                score_dict = self._get_score(iter_matrix)

                score = [word[0] for word in score_dict]
                score = score[:5]

                if len(score) < 5:
                    for i in range(5 - len(score)):
                        score.append('<PAD>')

            else:
                score = ['<PAD>' for _ in range(5)]

            if vocab is not None:   # 有vocab的时候需要进行变换为index
                out = [vocab.word_to_index(item) for item in score]
            else:
                out = score
            output.append(out)

        return output


    def word_tokenizer(self, sentence):
        out = self.tokenize_word(sentence)
        return out

    def rank_word(self, sentence, iter_num=None):
        if iter_num is not None:
            self.iter_num = iter_num

        # 分词并过滤一些词性
        word_list = self.filter_words(sentence)
        pairs_list = self._get_pairs(word_list)
        trans_matrix = self.get_matrix(pairs_list)
        init_matrix = np.ones((self.word_len, 1)) / self.word_len
        iter_matrix = init_matrix

        for _ in range(self.iter_num):
            iter_matrix = self.iter_matrix(trans_matrix, iter_matrix, init_matrix)

        score_dict = self._get_score(iter_matrix)
        return score_dict

    def topic_rank(self, sentence, topic=0, iter_num=None, word_only=True):
        if iter_num is not None:
            self.iter_num = iter_num
        word_list = self.filter_words(sentence)
        pairs_list = self._get_pairs(word_list)
        trans_matrix = self.get_matrix(pairs_list)

        iter_matrix = np.ones((self.word_len, 1)) / self.word_len
        init_matrix = np.zeros((self.word_len, 1))
        init_matrix[topic][0] = 1

        for _ in range(self.iter_num):
            iter_matrix = self.iter_matrix(trans_matrix, iter_matrix, init_matrix)

        score_dict = self._get_score(iter_matrix)
        if word_only is True:
            score = [word[0] for word in score_dict]
        else:
            score = score_dict
        return score


    def _get_score(self, matrix):
        score_dict = {}
        for index in range(self.word_len):
            score_dict[self.num2word[index]] = matrix[index][0]
        score_dict = sorted(score_dict.items(), key=lambda x: x[1], reverse=True)
        return score_dict

    def iter_matrix(self, trans_matrix, iter_matrix, init_matirx):
        out = self.alpha * np.dot(trans_matrix, iter_matrix) + (1 - self.alpha) * init_matirx
        return out

    def get_pos_tag(self, sentence, tagset='universal'):
        """
            获得句子序列的词性
        :param sentence: (str)
        :param tagset: split method
        :return:
        """
        token_words = self.tokenize_word(sentence)
        pos_words = pos_tag(token_words, tagset=tagset)
        return pos_words

    @staticmethod
    def tokenize_word(sentence):
        """
            分词
        :param sentence: (str)
        :return: (list)
        """
        token_words = tokenize.word_tokenize(sentence)
        token_words = [k.lower() for k in token_words]
        return token_words

    def filter_words(self, sentence, remain_tag=('NOUN', 'VERB', 'ADJ', 'ADV')):
        """
            得到过滤词性之后的单词，并创建词典
        """
        # remain_tag = ('NOUN', 'ADJ', 'ADV')
        result_word_list = []
        pos_words = self.get_pos_tag(sentence)
        out = self._filter_words(pos_words)
        return out

    def _filter_words(self, word_pos_pairs, remain_tag=('NOUN', 'VERB', 'ADJ', 'ADV')):
        out = []    # To obtain the remain words
        for word, tag in word_pos_pairs:
            if tag in remain_tag:
                out.append(word.lower())
        self._get_dict(out)
        return out

    def _get_pairs(self, word_list):
        """
            得到共现词对
        """
        word_num_list = [self._word2num[word] for word in word_list]
        cooc_pairs = []
        for i in range(len(word_num_list)):
            for j in range(1, self.span+1):
                if i + j < len(word_num_list):
                    cooc_pairs.append((word_num_list[i], word_num_list[i + j]))
                    cooc_pairs.append((word_num_list[i + j], word_num_list[i]))

        # TODO: weight is True
        return cooc_pairs

    def get_degree(self, sentence):
        remained_words = self.filter_words(sentence)
        word_pairs = self._get_pairs(remained_words)
        out = self.get_matrix(word_pairs, tensor=False, normalize=False)
        out = np.sum(out, axis=1)     # 计算每一个的度数
        max_num = np.max(out)         # 获取最大值
        min_num = np.min(out)         # 获取最小值

        # normalize度到 0~10 范围内
        # out = out * 10 / max_num
        out = (out - min_num) / (max_num - min_num + 1) * 10

        degree_dict = {'others': 0}
        for i in range(len(out)):
            degree_dict[self.num2word[i]] = int(out[i])

        out_degree = []

        for word in self.tokenize_word(sentence):
            if word in degree_dict:
                out_degree.append(degree_dict[word])
            else:
                out_degree.append(0)
        return out_degree

    def get_matrix(self, word_pairs_list, tensor=True, normalize=True):
        """
            得到共现矩阵
        """
        mat = np.zeros((self.word_len, self.word_len))  # 初始化全0矩阵
        for (index_0, index_1) in word_pairs_list:
            mat[index_0][index_1] = 1

        if normalize is True:
            edge_num = np.sum(mat, axis=0)
            mat = mat / edge_num
        # TODO: weight matrix
        if tensor is True:
            mat = torch.from_numpy(mat)
        return mat

    def _get_dict(self, word_list):
        words = set(word_list)
        self.word_len = len(words)
        self._word2num = {word: num for word, num in zip(words, range(self.word_len))}
        self.num2word = {num: word for word, num in zip(words, range(self.word_len))}

    def get_word_index(self, word):
        word = word.lower()
        return self._word2num[word]


if __name__ == '__main__':
    x = "Serge Ibaka -- the Oklahoma City Thunder forward who was born in the Congo " \
        "but played in Spain -- has been granted Spanish citizenship and will play for " \
        "the country in EuroBasket this summer, the event where spots in the 2012 Olympics " \
        "will be decided."
    y = "Serge Ibaka has been granted Spanish citizenship and will play " \
        "in EuroBasket"
    model = TextRank()
    out = model.get_degree(x)
    print(len(out))
    print(out)
    words = model.tokenize_word(x)
    print(len(words))
    for word, degree in zip(words, out):
        print(word, degree)

