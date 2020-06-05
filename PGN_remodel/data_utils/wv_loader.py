#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2020/3/7 16:10
# @Author : TheTAO
# @Site : 
# @File : wv_loader.py.py
# @Software: PyCharm
import numpy as np
from gensim.models.word2vec import Word2Vec
from PGN_remodel.data_utils.config import embedding_matrix_path, vocab_path, save_wv_model_path


class Vocab:
    PAD_TOKEN = '<PAD>'
    UNKNOWN_TOKEN = '<UNK>'
    START_DECODING = '<START>'
    STOP_DECODING = '<STOP>'
    # 这里将这些特殊符号统一定义为mask
    MASKS = [PAD_TOKEN, UNKNOWN_TOKEN, START_DECODING, STOP_DECODING]
    MASK_COUNT = len(MASKS)
    # mask_index
    PAD_TOKEN_INDEX = MASKS.index(PAD_TOKEN)
    UNKNOWN_TOKEN_INDEX = MASKS.index(UNKNOWN_TOKEN)
    START_DECODING_INDEX = MASKS.index(START_DECODING)
    STOP_DECODING_INDEX = MASKS.index(STOP_DECODING)

    def __init__(self, vocab_file=vocab_path, vocab_max_size=None):
        """
        Vocab 对象,vocab基本操作封装
        :param vocab_file: Vocab 存储路径
        :param vocab_max_size: 最大字典数量
        """
        self.word2id, self.id2word = self.load_vocab(vocab_file, vocab_max_size)
        self.count = len(self.word2id)

    def load_vocab(self, file_path, vocab_max_size=50000):
        """
        读取字典
        :param file_path: 文件路径
        :return: 返回读取后的字典
        """
        # 较之前不同的是此处的特殊符号index是手动添加的
        vocab = {mask: index for index, mask in enumerate(Vocab.MASKS)}
        reverse_vocab = {index: mask for index, mask in enumerate(Vocab.MASKS)}
        # 读取存储好的vocab文件
        for line in open(file_path, "r", encoding='utf-8').readlines():
            word, index = line.strip().split("\t")
            index = int(index)
            # 如果vocab 超过了指定大小
            # 跳出循环截断
            if vocab_max_size and index > vocab_max_size - Vocab.MASK_COUNT:
                print("最大的词典长度为 %i; 现在目前有%i词. 停止读取" % (vocab_max_size, index))
                break
            # 构造正反向词典 fix bug
            vocab[word] = index + Vocab.MASK_COUNT
            reverse_vocab[index + Vocab.MASK_COUNT] = word
        return vocab, reverse_vocab

    def word_to_id(self, word):
        if word not in self.word2id:
            return self.word2id[self.UNKNOWN_TOKEN]
        return self.word2id[word]

    def id_to_word(self, word_id):
        if word_id not in self.id2word:
            raise ValueError('Id not found in vocab: %d' % word_id)
        return self.id2word[word_id]

    def size(self):
        return self.count


# 较之前的此处将所有定义的mask符号作为0矩阵向量与词向量矩阵合并
def load_embedding_matrix(filepath=embedding_matrix_path, max_vocab_size=50000):
    """
    加载 embedding_matrix_path
    """
    embedding_matrix = np.load(filepath)
    flag_matrix = np.zeros_like(embedding_matrix[:Vocab.MASK_COUNT])
    return np.concatenate([flag_matrix, embedding_matrix])[:max_vocab_size]


def load_word2vec_file():
    # 保存词向量模型
    return Word2Vec.load(save_wv_model_path)


if __name__ == '__main__':
    # vocab 对象
    vocab = Vocab(vocab_path)
    print(vocab.word2id)
