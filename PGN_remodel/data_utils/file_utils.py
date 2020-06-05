#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2020/3/7 16:06
# @Author : TheTAO
# @Site : 
# @File : file_utils.py
# @Software: PyCharm
import re
import time
import pandas as pd
import numpy as np
from PGN_remodel.data_utils.config import *


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


# 存储词表
def save_vocab(file_path, data):
    with open(file_path, 'r', encoding='utf-8') as f:
        for i in data:
            f.write(i)


# 存储字典
def save_dict(file_path, dict_data):
    with open(file_path, 'w', encoding='utf-8') as f:
        for k, v in dict_data.items():
            f.write("{}\t{}\n".format(k, v))


# 加载字典
def load_dict(file_path):
    return dict((line.strip().split('/t')[0], idx) for idx, line in
                enumerate(open(file_path, 'r', encoding='utf-8').readlines()))


# 存储类似于用户字典的文件
def save_userdict(user_dict, file_path):
    with open(file_path, mode="w", encoding="utf-8") as f:
        f.write("\n".join(user_dict))


# 构造用户字典
def create_user_dict(*dataframe):
    """
    *dataframe，这样参数传递可以忽略个数
    创建自定义用户词典，主要是对Model，Brand中的词进行处理形成统一的用户字典
    :param dataframe: 传入的数据集
    :return:返回用户字典
    """

    # 处理掉形如“宝马X1(进口)”这样的词
    def process(sentence):
        r = re.compile(r"[(（]进口[)）]|\(海外\)|[^\u4e00-\u9fa5_a-zA-Z0-9]")
        return r.sub("", sentence)

    # 定义pandas中的Series
    user_dict = pd.Series()
    for df in dataframe:
        user_dict = pd.concat([user_dict, df.Model, df.Brand])
    # 去重操作
    user_dict = user_dict.apply(process).unique()
    # 删除空值
    user_dict = np.delete(user_dict, np.argwhere(user_dict == ""))
    return user_dict


# 根据网络参数去构造存储文件名
def get_result_filename(batch_size, epochs, embedding_dim, layer_units, commit=''):
    now_time = time.strftime('%Y_%m_%d_%H_%M_%S')
    file_name = now_time + 'batch_size_{}_epochs_{}_embedding_dim_{}_layerunits_{}_{}.csv'.format(
        batch_size, epochs,
        embedding_dim,
        layer_units,
        commit)
    result_save_path = os.path.join(save_result_dir, file_name)
    return result_save_path


if __name__ == '__main__':
    train_df, test_df = load_dataset(train_data_path, test_data_path)
    # 获取用户字典
    user_dict = create_user_dict(train_df, test_df)
    # 存储用户字典
    save_userdict(user_dict, '../datasets/cutDict/user_dict.txt')
