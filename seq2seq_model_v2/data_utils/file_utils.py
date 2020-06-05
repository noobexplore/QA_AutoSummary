#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2020/3/7 16:06
# @Author : TheTAO
# @Site : 
# @File : file_utils.py
# @Software: PyCharm
import os
import time
import pickle
from seq2seq_model_v2.data_utils.config import save_result_dir


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


# 根据网络参数去构造存储文件名
def get_result_filename(batch_size, epochs, max_length_inp, embedding_dim, layer_units, commit=''):
    now_time = time.strftime('%Y_%m_%d_%H_%M_%S')
    file_name = now_time + 'batch_size_{}_epochs_{}_max_length_inp_{}_embedding_dim_{}_layerunits_{}_{}.csv'.format(
        batch_size, epochs,
        max_length_inp,
        embedding_dim,
        layer_units,
        commit)
    result_save_path = os.path.join(save_result_dir, file_name)
    return result_save_path


def save_pickle(batch_data, pickle_path):
    f = open(pickle_path, 'wb')
    pickle.dump(batch_data, f)


def load_pickle(pickle_path):
    f = open(pickle_path, 'rb')
    batch_data = pickle.load(f)
    return batch_data