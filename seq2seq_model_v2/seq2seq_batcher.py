#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2020/3/7 15:57
# @Author : TheTAO
# @Site : 
# @File : batcher.py
# @Software: PyCharm
from seq2seq_model_v2.data_utils.params_utils import get_params
from seq2seq_model_v2.data_utils.data_loader import load_train_dataset, load_test_dataset
import tensorflow as tf

# 获取参数对象
params = get_params()


# 训练batch的生成函数
def train_batch_generator(batch_size, max_length_enc=params['max_enc_len'], max_length_dec=params['max_dec_len'],
                          sample_sum=None):
    # 加载数据集
    train_X, train_Y = load_train_dataset(max_length_enc, max_length_dec)
    # 采样总和
    if sample_sum:
        train_X = train_X[:sample_sum]
        train_Y = train_Y[:sample_sum]
    # 利用tf中的工具对数据集进行随机切片
    dataset = tf.data.Dataset.from_tensor_slices((train_X, train_Y)).shuffle(len(train_X))
    # 利用上面返回的对象进行取batch的操作
    dataset = dataset.batch(batch_size, drop_remainder=True)
    # 根据batch去计算跑完每个batch需要多少步
    steps_per_epoch = len(train_X) // batch_size
    return dataset, steps_per_epoch


# 同理进行测试batch的生成函数
def test_batch_generator(batch_size, max_length_enc=params['max_enc_len']):
    # 加载数据集
    test_X = load_test_dataset(max_length_enc)
    dataset = tf.data.Dataset.from_tensor_slices(test_X)
    dataset = dataset.batch(batch_size)
    steps_per_epoch = len(test_X) // batch_size
    return dataset, steps_per_epoch


# 用于beamsearch的批量生成器
def beam_test_batch_generator(beam_size, max_enc_len=params['max_enc_len']):
    # 加载数据集
    test_X = load_test_dataset(max_enc_len)
    for row in test_X:
        beam_search_data = tf.convert_to_tensor([row for i in range(beam_size)])
        yield beam_search_data


if __name__ == '__main__':
    for k, v in enumerate(beam_test_batch_generator(4)):
        print(k)
        print(v)