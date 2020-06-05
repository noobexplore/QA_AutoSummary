#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/5/27 10:55
# @Author  : TheTao
# @Site    : 
# @File    : data_loader2.py
# @Software: PyCharm
import jieba
import numpy as np
from gensim.models.word2vec import LineSentence, Word2Vec
from PGN_remodel.data_utils.multi_proc_utils import parallelize
from PGN_remodel.data_utils.file_utils import *
from PGN_remodel.data_utils.params_utils import *
from sklearn.model_selection import train_test_split
import warnings
# 引入日志配置
import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
warnings.filterwarnings("ignore")


def load_dataset(train_dataset_path, test_dataset_path):
    """
    加载数据集函数
    :param train_dataset_path:训练数据路径
    :param test_dataset_path:测试数据路径
    :return:train_df,test_df
    """
    train_df = pd.read_csv(train_dataset_path, encoding='utf-8')
    test_df = pd.read_csv(test_dataset_path, encoding='utf-8')
    print('train data size:{} and test data size:{}'.format(train_df.shape, test_df.shape))
    # 填充空值
    train_df.fillna('', inplace=True)
    test_df.fillna('', inplace=True)
    return train_df, test_df


def get_text(*dataframe, columns=["Question", "Dialogue", "Report"], concater=" "):
    """
    把训练集，测试集的文本拼接在一起，可用作以后训练语料使用
    :param dataframe: 传入一个包含数个df的元组
    :param columns: 要拼接的列
    :param concater: 怎么拼接列，默认用空格拼接
    :return:
    """
    text = ""
    for df in dataframe:
        # 过滤掉数据集没有的特征
        proc_columns = []
        for col in columns:
            if col in df.columns:
                proc_columns.append(col)
        # 把从第三列(包括)开始的数据拼在一起
        text += "\n".join(df[proc_columns].apply(lambda x: concater.join(x), axis=1))
    return text


# 加载停用词
def load_stop_words(stop_word_path):
    """
    加载停用词
    :param stop_word_path:停用词路径
    :return: 停用词表 list
    """
    # 打开文件
    file = open(stop_word_path, 'r', encoding='utf-8')
    # 读取所有行
    stop_words = file.readlines()
    # 去除每一个停用词前后 空格 换行符
    stop_words = [stop_word.strip() for stop_word in stop_words]
    return stop_words


def clean_single_sentence(sentence):
    """
    单句话进行清洗，包括句子中存在的特殊符号的处理
    :param sentence:待处理的句子
    :return:处理好的句子
    """
    # 首先过滤掉英文的标点改为中文的
    sentence = sentence.replace(",", "，")
    sentence = sentence.replace("!", "！")
    sentence = sentence.replace("?", "？")
    # 去掉形如1.和2.这类的标题
    r = re.compile(r"\D(\d\.)\D")
    sentence = r.sub("", sentence)
    # 删除形如（海外）等词
    r = re.compile(r"[(（]进口[)）]|\(海外\)")
    sentence = r.sub("", sentence)
    # 过滤掉除汉字字符和常用标点的所有字符
    r = re.compile(r"[^，！？。\.\-\u4e00-\u9fa5_a-zA-Z0-9]")
    sentence = r.sub("", sentence)
    # 删除车主说，技师说等大量重复的词
    r = re.compile(r"车主说|技师说|语音|图片|你好,|您好,|你好，|您好，")
    sentence = r.sub("", sentence)
    return sentence


# 加载停用词
stop_words_dict = load_stop_words(stop_word_path)


def filter_stopwords(words):
    """
    过滤停用词
    :param words:已经切好词的列表
    :return:过滤后的停用词
    """
    return [word for word in words if word not in stop_words_dict]


def sentence_proc(sentence):
    """
    利用jieba分词进行对中文句子分词，通过加载用户自定义词典
    :param sentence:待分词句子
    :return:分好词和过滤停用词的句子
    """
    # 先进行单句处理
    sentence = clean_single_sentence(sentence)
    # 采用全模式切词
    words = jieba.cut(sentence)
    # 再对其进行过滤停用词
    words = filter_stopwords(words)
    return " ".join(words)


def dataframe_proc(df):
    """
    数据批处理的方法
    :param df:传入的df
    :return:返回处理好的df
    """
    # jieba分词加载用户自定义词典
    jieba.load_userdict(user_dict)
    # 分别进行处理
    for col_name in ['Brand', 'Model', 'Question', 'Dialogue']:
        df[col_name] = df[col_name].apply(sentence_proc)
    # 训练集Report预处理
    if 'Report' in df.columns:
        df['Report'] = df['Report'].apply(sentence_proc)
    return df


def merge_traintest_dataframe(traindf, testdf):
    """
    合并有效信息并保存
    :param traindf:训练集df
    :param testdf:测试集df
    """
    print('合并有效信息')
    traindf['merged'] = traindf[['Question', 'Dialogue', 'Report']].apply(lambda x: ' '.join(x), axis=1)
    testdf['merged'] = testdf[['Question', 'Dialogue']].apply(lambda x: ' '.join(x), axis=1)
    mergeddf = pd.concat([traindf[['merged']], testdf[['merged']]], axis=0)
    # 5.将以上有效的数据存储为csv
    print('存储合并后的数据')
    traindf = traindf.drop(['merged'], axis=1)
    testdf = testdf.drop(['merged'], axis=1)
    traindf.to_csv(train_seg_path, index=None, encoding='utf-8')
    testdf.to_csv(test_seg_path, index=None, encoding='utf-8')
    # 6.保存合并数据
    mergeddf.to_csv(merger_seg_path, index=False, header=False, encoding='utf-8')


def train_word2vec(retrain=False):
    print('开始训练词向量')
    # 判断是否需要重新训练
    if not os.path.isfile(save_wv_model_path) or retrain:
        if retrain:
            print("重新", end="")
        # 如果没有保存的词向量，则开始训练
        wv_model = Word2Vec(LineSentence(merger_seg_path),
                            size=embedding_dim,
                            sg=1,  # skip_gram
                            workers=8,
                            iter=wv_train_epochs,
                            min_count=5,
                            seed=1)
        # 根据训练好的词向量模型去构建词典
        vocab = {word: index for index, word in enumerate(wv_model.wv.index2word)}
        reverse_vocab = {index: word for index, word in enumerate(wv_model.wv.index2word)}
        # 获取词向量矩阵
        embedding_matrix = wv_model.wv.vectors
        # 保存模型矩阵等操作
        print("词向量训练完毕，保存词向量模型、Embedding matrix和vocab")
        wv_model.save(save_wv_model_path)
        save_dict(vocab_path, vocab)
        save_dict(reverse_vocab_path, reverse_vocab)
        np.save(embedding_matrix_path, embedding_matrix)
    else:
        print("读取已训练好的词向量")
        wv_model = Word2Vec.load(save_wv_model_path)
    print('final w2v_model has vocabulary of ', len(wv_model.wv.vocab))
    return wv_model


def del_bad_sample(df, seg_df_path):
    """
    删除低质量样本，防止噪音问题
    :param df:主要是传入训练数据集
    :return:返回处理好的训练数据集
    """

    def detect_bad_words(x):
        for bad in bad_words:
            if (bad in x and len(x) <= 6):
                return True
        return False

    train = pd.read_csv(seg_df_path).fillna("")
    train["QD_nstr"] = train["Question"].apply(lambda x: len(x)) + train["Dialogue"].apply(lambda x: len(x))
    train["Rp_nstr"] = train["Report"].apply(lambda x: len(x))
    bad_words = ['参照下图', '参照图片', '参照图文', '参照图文',
                 '详见图片', '长时间不回复', '如图', '按图',
                 '看图', '见图', '随时联系', '已解决', '已经提供图片',
                 '已经发图片', '还在吗', '匹配']
    train["bad_words"] = train["Report"].apply(lambda x: detect_bad_words(x))
    train["car_master"] = train["Report"].apply(lambda x: "建议您下载汽车大师APP" in x)
    bad_sample_index = train[((train["QD_nstr"] >= 400) &  # Quesetion Dialogue 字符数>=400，且
                              (train["Rp_nstr"] <= 8)) |  # Report字符数<=8(882)，或
                             train["bad_words"] |  # 回复包括bad词(643)，或
                             (train["Rp_nstr"] < 2) |  # Report字符数<2(84)，或
                             train["car_master"]  # 回复推销汽车大师app(31)
                             ].index  # 共1482
    good_df = df.copy().drop(bad_sample_index, axis=0)
    print("共删除{}个低质量样本".format(len(bad_sample_index)))
    return good_df


def seg_traindata(traindf, testdf):
    print('开始划分数据集操作')
    # 划分训练集和标签数据集
    traindf['X'] = traindf[['Question', 'Dialogue']].apply(lambda x: ' '.join(x), axis=1)
    testdf['X'] = testdf[['Question', 'Dialogue']].apply(lambda x: ' '.join(x), axis=1)
    # 利用sklearn中的模块
    x_train, x_val, y_train, y_val = train_test_split(traindf['X'], traindf['Report'], test_size=0.002)
    x_train.to_csv(train_x_seg_path, index=None, header=False, encoding='utf-8')
    y_train.to_csv(train_y_seg_path, index=None, header=False, encoding='utf-8')
    x_val.to_csv(val_x_seg_path, index=None, header=False, encoding='utf-8')
    y_val.to_csv(val_y_seg_path, index=None, header=False, encoding='utf-8')
    # 转化为csv文件进行存储
    traindf['X'].to_csv(test_x_seg_path, index=None, header=False, encoding='utf-8')
    testdf['X'].to_csv(test_x_seg_path, index=None, header=False, encoding='utf-8')
    return traindf, testdf


def get_max_len(data):
    """
    获得合适的最大长度值
    :param data: 待统计的数据  train_df['Question']
    :return: 最大长度值
    """
    max_lens = data.apply(lambda x: x.count(' ') + 1)
    return int(np.mean(max_lens) + 2 * np.std(max_lens))


def train_test_x_max_len(traindf, testdf):
    print('计算合适的长度')
    # 计算出合适的长度
    train_x_max_len = get_max_len(traindf['X'])
    test_x_max_len = get_max_len(testdf['X'])
    x_max_len = max(train_x_max_len, test_x_max_len)
    y_max_len = get_max_len(train_df['Report'])
    return x_max_len, y_max_len


def pad_proc(sentence, max_len, vocab):
    """
    填充字段 < start > < end > < pad > < unk > max_lens
    :param sentence: 处理后的句子
    :param max_len: 最大长度
    :param vocab: 词典
    :return:
    """
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


def pad_df(traindf, testdf, max_xlen, max_ylen, vocab):
    print('开始填充数据')
    # 分别进行填充特殊字符
    traindf['X'] = traindf['X'].apply(lambda x: pad_proc(x, max_xlen, vocab))
    testdf['X'] = testdf['X'].apply(lambda x: pad_proc(x, max_xlen, vocab))
    traindf['Y'] = traindf['Report'].apply(lambda x: pad_proc(x, max_ylen, vocab))
    # 然后分别进行保存
    print('保存填充数据')
    traindf['X'].to_csv(train_x_pad_path, index=None, header=False, encoding='utf-8')
    traindf['Y'].to_csv(train_y_pad_path, index=None, header=False, encoding='utf-8')
    testdf['X'].to_csv(test_x_pad_path, index=None, header=False, encoding='utf-8')
    print('train_x_max_len:{} ,train_y_max_len:{}'.format(max_xlen, max_ylen))
    return traindf, testdf


def transform_data(sentence, vocab):
    """
    转化为数字索引函数
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


def transform_df(traindf, testdf, vocab):
    print('数字索引转化')
    train_ids_x = traindf['X'].apply(lambda x: transform_data(x, vocab))
    train_ids_y = traindf['Y'].apply(lambda x: transform_data(x, vocab))
    test_ids_x = testdf['X'].apply(lambda x: transform_data(x, vocab))
    print('转化为numpy数组并保存')
    train_X = np.array(train_ids_x.tolist())
    train_Y = np.array(train_ids_y.tolist())
    test_X = np.array(test_ids_x.tolist())
    # 17.最后保存numpy数组
    np.save(train_x_path, train_X)
    np.save(train_y_path, train_Y)
    np.save(test_x_path, test_X)
    return train_X, train_Y, test_X


# 加载处理好的数据集
def load_processed_set(max_enc_len, max_dec_len):
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


if __name__ == '__main__':
    start_time = time.time()  # 计时开始
    # 加载数据集
    train_df, test_df = load_dataset(train_data_path, test_data_path)
    # 获取拼接文本
    raw_text = get_text(train_df, test_df)
    # 并行处理合理的利用多个cpu进行加速运算
    train_df = parallelize(train_df, dataframe_proc)
    test_df = parallelize(test_df, dataframe_proc)
    # 合并有效信息并存储
    merge_traintest_dataframe(train_df, test_df)
    # 开始训练词向量
    wv_model = train_word2vec()
    # 去除掉train中质量较低的数据
    train_df = del_bad_sample(train_df, train_seg_path)
    # 分开训练集和标签
    train_df, test_df = seg_traindata(train_df, test_df)
    # 获取到合适的长度
    x_max_len, y_max_len = train_test_x_max_len(train_df, test_df)
    # 更新params中的max_len值
    # 加载参数管理器
    params = get_params()
    params['max_enc_len'] = x_max_len
    params['max_dec_len'] = y_max_len
    # 加载字典
    Vocab = Vocab()
    vocab, _ = Vocab.load_vocab(file_path=vocab_path)
    # 进行填充操作
    train_df, test_df = pad_df(train_df, test_df, x_max_len, y_max_len, vocab=vocab)
    # 数字索引的转化
    train_X, train_Y, test_X = transform_df(train_df, test_df, vocab)
    print("共耗时{:.2f}s".format(time.time() - start_time))
