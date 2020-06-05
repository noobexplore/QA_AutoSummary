#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/5/28 11:55
# @Author  : TheTao
# @Site    : 
# @File    : pgn_train_helper.py
# @Software: PyCharm
import os
import time
import tensorflow as tf
from PGN_remodel.bacher import batcher
from PGN_remodel.loss import calc_loss
from PGN_remodel.data_utils.config import checkpoint_dir


def get_train_msg(ckpt):
    # 获得已训练的轮次
    path = os.path.join(ckpt, "trained_epoch.txt")
    with open(path, mode="r", encoding="utf-8") as f:
        trained_epoch = int(f.read())
    return trained_epoch


def save_train_msg(ckpt, trained_epoch):
    # 保存训练信息（已训练的轮数）
    path = os.path.join(ckpt, "trained_epoch.txt")
    with open(path, mode="w", encoding="utf-8") as f:
        f.write(str(trained_epoch))


# 开始训练
def train_model(model, vocab, params, checkpoint_manager):
    epochs = params['epochs']
    # 初始化优化器
    optimizer = tf.keras.optimizers.Adagrad(params['learning_rate'],
                                            initial_accumulator_value=params['adagrad_init_acc'],
                                            clipnorm=params['max_grad_norm'],
                                            epsilon=params['eps'])

    # 训练步数
    def train_step(target, enc_inp, dec_inp, enc_extended_inp, batch_oov_len, enc_mask, dec_mask, cov_loss_wt):
        # 拿到梯度计算loss
        with tf.GradientTape() as tape:
            final_dist, attentions, coverages \
                = model(enc_inp, dec_inp, enc_extended_inp, batch_oov_len, enc_mask)

            batch_loss, log_loss, cov_loss \
                = calc_loss(target, final_dist, dec_mask, attentions, coverages, cov_loss_wt)

        variables = model.trainable_variables
        gradients = tape.gradient(batch_loss, variables)
        optimizer.apply_gradients(zip(gradients, variables))
        return batch_loss, log_loss, cov_loss

    dataset = batcher(vocab, params)
    steps_per_epoch = params["steps_per_epoch"]

    # 开始迭代训练
    for epoch in range(epochs):
        start = time.time()
        total_loss = total_log_loss = total_cov_loss = 0
        for (batch, (enc_data, dec_data)) in enumerate(dataset.take(steps_per_epoch)):
            # 以防万一，传进去的参数全为tensor
            cov_loss_wt = tf.cast(params["cov_loss_wt"], dtype=tf.float32)
            try:
                batch_oov_len = tf.shape(enc_data["article_oovs"])[1]
            except:
                batch_oov_len = tf.constant(0)
            # 拿到每一步的loss
            batch_loss, log_loss, cov_loss = train_step(dec_data["dec_target"],
                                                        enc_data["enc_input"],
                                                        dec_data["dec_input"],
                                                        enc_data["extended_enc_input"],
                                                        batch_oov_len,
                                                        enc_data["enc_mask"],
                                                        dec_data["dec_mask"],
                                                        cov_loss_wt)
            # 计算总的loss
            total_loss += batch_loss
            total_log_loss += log_loss
            total_cov_loss += cov_loss
            if (batch + 1) % 5 == 0:
                print('Epoch {} Batch {} batch_loss {:.4f} log_loss {:.4f} cov_loss {:.4f}'.format(
                    params["trained_epoch"] + epoch + 1,
                    batch + 1,
                    batch_loss.numpy(),
                    log_loss.numpy(),
                    cov_loss.numpy()))
        # saving (checkpoint) the model every 2 epochs
        if (epoch + 1) % 1 == 0:
            ckpt_save_path = checkpoint_manager.save()
            try:
                record_file = os.path.join(checkpoint_dir, "record.txt")
                with open(record_file, mode="a", encoding="utf-8") as f:
                    f.write('Epoch {} Loss {:.4f}\n'.format(params["trained_epoch"] + epoch + 1,
                                                            total_loss / steps_per_epoch))
            except:
                pass
            print('Saving checkpoint for epoch {} at {}'.format(epoch + 1, ckpt_save_path))

            # ---------------学习率衰减---------------------------------------------
            # lr = params["learning_rate"] * np.power(0.95, epoch + 1)
            # # 更新优化器的学习率
            # optimizer = tf.keras.optimizers.Adam(name='Adam', learning_rate=lr)
            # assert lr == optimizer.get_config()["learning_rate"]
            # print("learning_rate=", optimizer.get_config()["learning_rate"])
            # ---------------------------------------------------------------------

            save_train_msg(checkpoint_dir, params["trained_epoch"] + epoch + 1)  # 保存已训练的轮数
        # 打印信息
        print('Epoch {} Loss {:.4f} log_loss {:.4f} cov_loss {:.4f}'.format(
            params["trained_epoch"] + epoch + 1,
            total_loss / steps_per_epoch,
            total_log_loss / steps_per_epoch,
            total_cov_loss / steps_per_epoch))
        print('Time taken for 1 epoch {} sec\n'.format(time.time() - start))
