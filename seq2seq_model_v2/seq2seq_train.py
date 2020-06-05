#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2020/3/7 17:30
# @Author : TheTAO
# @Site : 
# @File : seq2seq_train.py
# @Software: PyCharm
import tensorflow as tf
from seq2seq_model_v2.seq2seq_model import Seq2Seq
from seq2seq_model_v2.seq2seq_train_helper import train_model
from seq2seq_model_v2.data_utils.gpu_utils import config_gpu
from seq2seq_model_v2.data_utils.params_utils import get_params
from seq2seq_model_v2.data_utils.wv_loader import Vocab


# 训练函数
def train(params):
    # GPU资源配置
    config_gpu()
    # 读取vocab
    vocab = Vocab(params["vocab_path"], params["vocab_size"])
    # 读取vocab_size
    params['vocab_size'] = vocab.count
    # 首先初始化构建模型
    print("初始化构建模型 ...")
    model = Seq2Seq(params)
    # 获取模型保存管理对象
    print("读取训练好的模型 ...")
    checkpoint = tf.train.Checkpoint(Seq2Seq=model)
    checkpoint_manager = tf.train.CheckpointManager(checkpoint, params['checkpoint_dir'], max_to_keep=5)
    print("开始训练 ...")
    # 开始训练
    train_model(model, vocab, params, checkpoint_manager)


if __name__ == '__main__':
    # 获得参数
    params = get_params()
    # 训练模型
    train(params)
