#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2020/3/7 16:12
# @Author : TheTAO
# @Site : 
# @File : data_loader.py
# @Software: PyCharm
import re
import jieba
import numpy as np
import pandas as pd
from PGN_model.data_utils.wv_loader import Vocab
from PGN_model.data_utils.params_utils import get_params
from PGN_model.data_utils.file_utils import save_dict
from PGN_model.data_utils.multi_proc_utils import parallelize
from PGN_model.data_utils.config import *
from gensim.models.word2vec import LineSentence, Word2Vec
from PGN_model.data_utils.config import save_wv_model_path
from sklearn.model_selection import train_test_split
# 引入日志配置
import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# 加载自定义词表
jieba.load_userdict(user_dict)
# 加载参数管理器
params = get_params()


def build_dataset(train_data_path, test_data_path):
    """
    数据加载和预处理
    :param train_data_path:训练数据集路径
    :param test_data_path:测试数据集路径
    :return:训练数据、测试、合并后的数据
    """
    # 1.先加载数据
    print('第一步：加载训练和测试数据集')
    train_df = pd.read_csv(train_data_path, encoding='utf-8')
    test_df = pd.read_csv(test_data_path, encoding='utf-8')
    print('train data size:{} and test data size:{}'.format(train_df.shape, test_df.shape))
    print('Dataset is complete')
    # 2.空值的填充 利用pandas里面的dropna函数
    print('第二步：空值的填充')
    train_df.dropna(subset=['Report'], inplace=True)
    train_df.fillna('', inplace=True)
    test_df.fillna('', inplace=True)
    # 3.将一些预处理操作进行预处理操作，a.噪声清洗 b.过滤停用词 c.分词处理
    print('第三步：多线程操作')
    train_df = parallelize(train_df, sentences_proc)
    test_df = parallelize(test_df, sentences_proc)
    # 4.表有效内容的合并方便后续的处理
    print('第四步：合并有效信息')
    train_df['merged'] = train_df[['Question', 'Dialogue', 'Report']].apply(lambda x: ' '.join(x), axis=1)
    test_df['merged'] = test_df[['Question', 'Dialogue']].apply(lambda x: ' '.join(x), axis=1)
    merged_df = pd.concat([train_df[['merged']], test_df[['merged']]], axis=0)
    print('train data size {},test data size {},merged_df data size {}'.format(len(train_df), len(test_df),
                                                                               len(merged_df)))
    # 5.将以上有效的数据存储为csv
    print('第五步：存储合并后的数据')
    train_df = train_df.drop(['merged'], axis=1)
    test_df = test_df.drop(['merged'], axis=1)
    train_df.to_csv(train_seg_path, index=None, header=False, encoding='utf-8')
    test_df.to_csv(test_seg_path, index=None, header=False, encoding='utf-8')
    # 6.保存合并数据
    merged_df.to_csv(merger_seg_path, index=None, header=False, encoding='utf-8')
    # 7.训练词向量
    print('第六步：开始构建词向量训练')
    wv_model = Word2Vec(LineSentence(merger_seg_path), size=embedding_dim, sg=1, workers=8,
                        iter=wv_train_epochs, window=5, min_count=5)
    # 8.分离训练数据和标签数据
    print('第七步：分离训练数据和标签数据')
    train_df['X'] = train_df[['Question', 'Dialogue']].apply(lambda x: ' '.join(x), axis=1)
    test_df['X'] = test_df[['Question', 'Dialogue']].apply(lambda x: ' '.join(x), axis=1)
    # 9.训练集和验证集的划分（这里与之前有改动）
    print('第八步：划分训练集和测试集')
    # 8W*0.002
    X_train, X_val, y_train, y_val = train_test_split(train_df['X'], train_df['Report'], test_size=0.002)
    # 转化为csv文件
    X_train.to_csv(train_x_seg_path, index=None, header=False, encoding='utf-8')
    y_train.to_csv(train_y_seg_path, index=None, header=False, encoding='utf-8')
    X_val.to_csv(val_x_seg_path, index=None, header=False, encoding='utf-8')
    y_val.to_csv(val_y_seg_path, index=None, header=False, encoding='utf-8')
    test_df['X'].to_csv(test_x_seg_path, index=None, header=False, encoding='utf-8')
    # 10.各类的填充操作，包括填充开始结束符，oov填充词，长度填充
    print('第九步：开始填充操作')
    # 取出训练好的词向量表
    vocab = wv_model.wv.vocab
    # 获取适当的长度
    train_x_max_len = get_max_len(train_df['X'])
    test_X_max_len = get_max_len(test_df['X'])
    X_max_len = max(train_x_max_len, test_X_max_len)
    # 训练集的填充操作
    train_df['X'] = train_df['X'].apply(lambda x: pad_proc(x, X_max_len, vocab))
    # 测试集的填充操作
    test_df['X'] = test_df['X'].apply(lambda x: pad_proc(x, X_max_len, vocab))
    # 标签数据的同样处理
    train_y_max_len = get_max_len(train_df['Report'])
    train_df['Y'] = train_df['Report'].apply(lambda x: pad_proc(x, train_y_max_len, vocab))
    # 11.保存以上的填充数据
    print('第十步：保存填充数据')
    train_df['X'].to_csv(train_x_pad_path, index=None, header=False, encoding='utf-8')
    train_df['Y'].to_csv(train_y_pad_path, index=None, header=False, encoding='utf-8')
    test_df['X'].to_csv(test_x_pad_path, index=None, header=False, encoding='utf-8')
    print('train_x_max_len:{} ,train_y_max_len:{}'.format(X_max_len, train_y_max_len))
    # 较之前的此处就不再从新的训练特殊符号的词向量因为后面都会视为mask，从而不会去计算这一部分所以重新训练没有意义
    # 12.再次的训练词向量，因为新填充的特殊字符也需要得到词向量矩阵
    # 保存重新训练的词向量
    wv_model.save(save_wv_model_path)
    print('finish retrain w2v model')
    print('final w2v_model has vocabulary of ', len(wv_model.wv.vocab))
    # 13.更新词典
    print('第十二步：开始更新词典')
    vocab = {word: index for index, word in enumerate(wv_model.wv.index2word)}
    reverse_vocab = {index: word for index, word in enumerate(wv_model.wv.index2word)}
    # 保存字典
    save_dict(vocab_path, vocab)
    save_dict(reverse_vocab_path, reverse_vocab)
    # 14.保存词向量矩阵
    print('第十三步：保存词向量矩阵')
    embedding_matrix = wv_model.wv.vectors
    np.save(embedding_matrix_path, embedding_matrix)
    # 15.数据集的批量转换为索引形式义工后面训练
    print('第十四步：将数据转化为索引')
    train_ids_x = train_df['X'].apply(lambda x: transform_data(x, vocab))
    train_ids_y = train_df['Y'].apply(lambda x: transform_data(x, vocab))
    test_ids_x = test_df['X'].apply(lambda x: transform_data(x, vocab))
    # 16.再将其以上数据转化为numpy数组
    print('第十五步：转化为numpy数组并保存')
    train_X = np.array(train_ids_x.tolist())
    train_Y = np.array(train_ids_y.tolist())
    test_X = np.array(test_ids_x.tolist())
    # 17.最后保存numpy数组
    np.save(train_x_path, train_X)
    np.save(train_y_path, train_Y)
    np.save(test_x_path, test_X)
    return train_X, train_Y, test_X


# 单句话进行的切词填充和转换操作
def preprocess_sentence(sentence, max_len, vocab):
    # 1. 切词处理
    sentence = sentence_proc(sentence)
    # 2. 填充
    sentence = pad_proc(sentence, max_len - 2, vocab)
    # 3. 转换index
    sentence = transform_data(sentence, vocab)
    return np.array([sentence])


# 加载处理好的数据集
def load_dataset(max_enc_len=params['max_enc_len'], max_dec_len=params['max_dec_len']):
    train_X = np.load(train_x_path + '.npy')
    train_Y = np.load(train_y_path + '.npy')
    test_X = np.load(test_x_path + '.npy')
    # 根据最大输入长度去截取
    train_X = train_X[:, :max_enc_len]
    train_Y = train_Y[:, :max_dec_len]
    test_X = test_X[:, :max_enc_len]
    # 返回处理好的数据集
    return train_X, train_Y, test_X


# 单独加载训练数据以及标签数据
def load_train_dataset(max_enc_len, max_dec_len):
    train_X = np.load(train_x_path + '.npy')
    train_Y = np.load(train_y_path + '.npy')

    train_X = train_X[:, :max_enc_len]
    train_Y = train_Y[:, :max_dec_len]
    return train_X, train_Y


# 单独加载测试数据
def load_test_dataset(max_enc_len):
    test_X = np.load(test_x_path + '.npy')
    test_X = test_X[:, :max_enc_len]
    return test_X


# 根据公式获取期望的最大长度
def get_max_len(data):
    """
    获得合适的最大长度值
    :param data: 待统计的数据  train_df['Question']
    :return: 最大长度值
    """
    # TODO FIX len size bug
    max_lens = data.apply(lambda x: x.count(' ') + 1)
    return int(np.mean(max_lens) + 2 * np.std(max_lens))


def transform_data(sentence, vocab):
    """
    word 2 index
    :param sentence: [word1,word2,word3, ...] ---> [index1,index2,index3 ......]
    :param vocab: 词表
    :return: 转换后的序列
    """
    # 字符串切分成词
    words = sentence.split(' ')
    # 按照vocab的index进行转换, 遇到未知词就填充unk的索引
    ids = [vocab[word] if word in vocab else Vocab.UNKNOWN_TOKEN_INDEX for word in words]
    return ids


def pad_proc(sentence, max_len, vocab):
    '''
    # 填充字段
    < start > < end > < pad > < unk > max_lens
    '''
    # 0.按空格统计切分出词
    words = sentence.strip().split(' ')
    # 1. 截取规定长度的词数
    words = words[:max_len]
    # 2. 填充< unk > ,判断是否在vocab中, 不在填充 < unk >
    sentence = [word if word in vocab else Vocab.UNKNOWN_TOKEN for word in words]
    # 3. 填充< start > < end >
    sentence = [Vocab.START_DECODING] + sentence + [Vocab.STOP_DECODING]
    # 4. 判断长度，填充< pad >
    sentence = sentence + [Vocab.PAD_TOKEN] * (max_len - len(words))
    return ' '.join(sentence)


# 加载停用词
def load_stop_words(stop_word_path):
    '''
    加载停用词
    :param stop_word_path:停用词路径
    :return: 停用词表 list
    '''
    # 打开文件
    file = open(stop_word_path, 'r', encoding='utf-8')
    # 读取所有行
    stop_words = file.readlines()
    # 去除每一个停用词前后 空格 换行符
    stop_words = [stop_word.strip() for stop_word in stop_words]
    return stop_words


# 加载停用词
stop_words = load_stop_words(stop_word_path)
# 额外需要移除的词
remove_words = ['|', '[', ']', '语音', '图片', '车主说', '技师说']


# 特殊符号的清洗及是一些噪声符号
def clean_sentence(sentence):
    '''
    特殊符号去除
    :param sentence: 待处理的字符串
    :return: 过滤特殊字符后的字符串
    '''
    if isinstance(sentence, str):
        return re.sub(
            r'[\s+\-\/\[\]\{\}_$%^*(+\"\')]+|[+——()【】“”~@#￥%……&*（）]+|你好,|您好,|你好，|您好，',
            ' ', sentence)
    else:
        return ' '


# 过滤停用词
def filter_words(sentence):
    '''
    过滤停用词
    :param seg_list: 切好词的列表 [word1 ,word2 .......]
    :return: 过滤后的停用词
    '''
    words = sentence.split(' ')
    # 去掉多余空字符
    words = [word for word in words if word and word not in remove_words]
    # 去掉停用词 包括一下标点符号也会去掉
    words = [word for word in words if word not in stop_words]
    return words


# 去除|的影响进行切分
def seg_proc(sentence):
    tokens = sentence.split('|')
    result = []
    for t in tokens:
        result.append(cut_sentence(t))
    return ' | '.join(result)


# 切词函数
def cut_sentence(line):
    # 切词，默认精确模式，全模式cut参数cut_all=True
    tokens = jieba.cut(line)
    return ' '.join(tokens)


# 单个句子预处理函数
def sentence_proc(sentence):
    '''
    预处理模块
    :param sentence:待处理字符串
    :return: 处理后的字符串
    '''
    # 清除无用词
    sentence = clean_sentence(sentence)
    # 分段切词
    sentence = seg_proc(sentence)
    # 过滤停用词
    words = filter_words(sentence)
    # 拼接成一个字符串,按空格分隔
    return ' '.join(words)


# 结合到数据表的预处理函数
def sentences_proc(df):
    '''
    数据集批量处理方法
    :param df: 数据集
    :return:处理好的数据集
    '''
    # 批量预处理 训练集和测试集
    for col_name in ['Brand', 'Model', 'Question', 'Dialogue']:
        df[col_name] = df[col_name].apply(sentence_proc)

    if 'Report' in df.columns:
        # 训练集 Report 预处理
        df['Report'] = df['Report'].apply(sentence_proc)
    return df


if __name__ == '__main__':
    # 数据集批量处理
    build_dataset(train_data_path, test_data_path)
