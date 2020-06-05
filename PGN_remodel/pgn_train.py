#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/5/28 14:16
# @Author  : TheTao
# @Site    : 
# @File    : pgn_train.py
# @Software: PyCharm
import os
import pathlib
import numpy as np
import tensorflow as tf
from PGN_remodel.PGN_model import PGN
from PGN_remodel.pgn_train_helper import train_model, get_train_msg
from PGN_remodel.data_utils.gpu_utils import config_gpu
from PGN_remodel.data_utils.params_utils import get_params
from PGN_remodel.data_utils.wv_loader import Vocab
from PGN_remodel.data_utils.config import checkpoint_dir


def train(params):
    # GPU资源配置
    # config_gpu()
    # 读取vocab训练
    vocab = Vocab(params["vocab_path"], params["vocab_size"])
    params['checkpoint_dir'] = checkpoint_dir
    params['vocab_size'] = vocab.count
    params["trained_epoch"] = get_train_msg(checkpoint_dir)
    # 学习率衰减
    params["learning_rate"] *= np.power(0.95, params["trained_epoch"])
    # 构建模型
    print("Building the model ...")
    model = PGN(params)
    # 获取保存管理者
    checkpoint = tf.train.Checkpoint(PGN=model)
    checkpoint_manager = tf.train.CheckpointManager(checkpoint, checkpoint_dir, max_to_keep=5)
    checkpoint.restore(checkpoint_manager.latest_checkpoint)
    if checkpoint_manager.latest_checkpoint:
        print("Restored from {}".format(checkpoint_manager.latest_checkpoint))
    else:
        print("Initializing from scratch.")
    # 开始训练模型
    print("开始训练模型..")
    print("trained_epoch:", params["trained_epoch"])
    print("mode:", params["mode"])
    print("epochs:", params["epochs"])
    print("batch_size:", params["batch_size"])
    print("max_enc_len:", params["max_enc_len"])
    print("max_dec_len:", params["max_dec_len"])
    print("learning_rate:", params["learning_rate"])
    train_model(model, vocab, params, checkpoint_manager)


if __name__ == '__main__':
    params = get_params()
    train(params)
