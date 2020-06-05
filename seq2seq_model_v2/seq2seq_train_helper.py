#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2020/3/7 17:18
# @Author : TheTAO
# @Site : 
# @File : seq2seq_train_helper.py
# @Software: PyCharm
import time
import tensorflow as tf
from seq2seq_model_v2.seq2seq_batcher import train_batch_generator


def train_model(model, vocab, params, checkpoint_manager):
    # 初始化一些参数
    epochs = params['epochs']
    batch_size = params['batch_size']
    pad_index = vocab.word2id[vocab.PAD_TOKEN]
    start_index = vocab.word2id[vocab.START_DECODING]
    # 获取vocab_size
    params['vocab_size'] = vocab.count
    # 优化器以及损失函数的定义
    optimizer = tf.keras.optimizers.Adam(name='Adam', learning_rate=0.0001)
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')

    # 定义损失函数
    def loss_function(real, pred):
        # 首先定义mask矩阵以便于去计算loss，去掉为0标签的loss
        mask = tf.math.logical_not(tf.math.equal(real, pad_index))
        loss_ = loss_object(real, pred)
        mask = tf.cast(mask, dtype=loss_.dtype)
        loss_ *= mask
        return tf.reduce_mean(loss_)

    # 训练
    @tf.function
    def train_step(enc_inp, dec_target):
        batch_loss = 0
        with tf.GradientTape() as tape:
            enc_output, enc_hidden = model.call_encoder(enc_inp)
            # 第一个decoder输入 开始标签
            dec_input = tf.expand_dims([start_index] * batch_size, 1)
            # 第一个隐藏层输入
            dec_hidden = enc_hidden
            # 逐个预测序列
            predictions, _ = model(dec_input, dec_hidden, enc_output, dec_target)
            # 计算每个batch的loss
            batch_loss = loss_function(dec_target[:, 1:], predictions)
            variables = model.encoder.trainable_variables + model.decoder.trainable_variables + model.attention.trainable_variables
            gradients = tape.gradient(batch_loss, variables)
            optimizer.apply_gradients(zip(gradients, variables))
            return batch_loss

    # 获取数据集
    dataset, steps_per_epoch = train_batch_generator(batch_size)
    # 迭代每一步
    for epoch in range(epochs):
        start = time.time()
        total_loss = 0
        for (batch, (inputs, target)) in enumerate(dataset.take(steps_per_epoch)):
            batch_loss = train_step(inputs, target)
            total_loss += batch_loss
            if batch % 10 == 0:
                print('Epoch {} Batch {} Loss {:.4f}'.format(epoch + 1, batch, batch_loss.numpy()))
        # 每5个epoch存储一次
        if (epoch + 1) % 5 == 0:
            ckpt_save_path = checkpoint_manager.save()
            print('Saving checkpoint for epoch {} at {}'.format(epoch + 1, ckpt_save_path))
        print('Epoch {} total_Loss {:.4f}'.format(epoch + 1, total_loss / steps_per_epoch))
        print('Time taken for 1 epoch {} sec\n'.format(time.time() - start))
