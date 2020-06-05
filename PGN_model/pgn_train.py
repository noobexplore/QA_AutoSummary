#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2020/3/10 10:52
# @Author : TheTAO
# @Site : 
# @File : pgn_train.py
# @Software: PyCharm
import tensorflow as tf
from PGN_model.data_utils.gpu_utils import config_gpu
from PGN_model.batcher import batcher
from PGN_model.pgn_model import PGN
from PGN_model.pgn_train_helper import train_model
from PGN_model.data_utils.params_utils import get_params
from PGN_model.data_utils.wv_loader import Vocab


def train(params):
    # GPU资源配置
    config_gpu(use_cpu=False, gpu_memory=params['gpu_memory'])
    # 读取vocab训练
    print("Building the model ...")
    vocab = Vocab(params["vocab_path"], params["max_vocab_size"])
    params['vocab_size'] = vocab.count
    print("词典大小为：{}".format(vocab.count))
    # 构建模型
    print("Building the model ...")
    model = PGN(params)
    print("Creating the batcher ...")
    dataset = batcher(vocab, params)
    # 获取保存管理者
    print("Creating the checkpoint manager")
    checkpoint = tf.train.Checkpoint(PGN=model)
    checkpoint_manager = tf.train.CheckpointManager(checkpoint, params['checkpoint_dir'], max_to_keep=5)
    checkpoint.restore(checkpoint_manager.latest_checkpoint)
    if checkpoint_manager.latest_checkpoint:
        print("Restored from {}".format(checkpoint_manager.latest_checkpoint))
    else:
        print("Initializing from scratch.")
    # 训练模型
    print("Starting the training ...")
    train_model(model, dataset, params, checkpoint_manager)


if __name__ == '__main__':
    # 获得参数m
    params = get_params()
    params['mode'] = 'train'
    params['max_enc_len'] = 200
    params['max_dec_len'] = 50
    params['batch_size'] = 1
    params['pointer_gen'] = True
    params['use_coverage'] = True
    params['max_vocab_size'] = 50000
    params['cov_loss_wt'] = 1.0
    # 训练模型
    train(params)
