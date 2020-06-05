#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2020/3/9 19:17
# @Author : TheTAO
# @Site : 
# @File : train_helper.py
# @Software: PyCharm
import time
import tensorflow as tf
import numpy as np
from PGN_model.loss import calc_loss


def train_model(model, dataset, params, checkpoint_manager):
    epochs = params['epochs']
    optimizer = tf.keras.optimizers.Adagrad(params['learning_rate'],
                                            initial_accumulator_value=params['adagrad_init_acc'],
                                            clipnorm=params['max_grad_norm'],
                                            epsilon=params['eps'])

    @tf.function(input_signature=(tf.TensorSpec(shape=[params["batch_size"], None], dtype=tf.int32),
                                  tf.TensorSpec(shape=[params["batch_size"], None], dtype=tf.int32),
                                  tf.TensorSpec(shape=[], dtype=tf.int32),
                                  tf.TensorSpec(shape=[params["batch_size"], params["max_dec_len"]], dtype=tf.int32),
                                  tf.TensorSpec(shape=[params["batch_size"], params["max_dec_len"]], dtype=tf.int32),
                                  tf.TensorSpec(shape=[params["batch_size"], None], dtype=tf.int32),
                                  tf.TensorSpec(shape=[params["batch_size"], params["max_dec_len"]], dtype=tf.int32)))
    def train_step(enc_inp, extended_enc_input, max_oov_len, dec_input, dec_target, enc_pad_mask, dec_pad_mask):
        # 这样方便拿到每一步的梯度
        with tf.GradientTape() as tape:
            # 还是先encoder
            enc_output, enc_hidden = model.call_encoder(enc_inp)
            # 然后初始化dec_hidden
            dec_hidden = enc_hidden
            # 再计算final_dist, attentions, coverages
            final_dists, _, attentions, coverages = model(dec_hidden,
                                                          enc_output,
                                                          dec_input,
                                                          extended_enc_input,
                                                          max_oov_len,
                                                          enc_pad_mask=enc_pad_mask,
                                                          use_coverage=params['use_coverage'],
                                                          prev_coverage=None)
            # 计算loss
            batch_loss, log_loss, cov_loss = calc_loss(dec_target, final_dists, dec_pad_mask, attentions, coverages,
                                                       params['cov_loss_wt'],
                                                       params['use_coverage'],
                                                       params['pointer_gen'])
        # 获取训练的参数
        variables = model.encoder.trainable_variables + model.decoder.trainable_variables + \
                    model.attention.trainable_variables + model.pointer.trainable_variables
        # 更新梯度
        gradients = tape.gradient(batch_loss, variables)
        optimizer.apply_gradients(zip(gradients, variables))
        return batch_loss, log_loss, cov_loss

    # 开始迭代训练
    best_loss = 10
    for epoch in range(epochs):
        start = time.time()
        total_loss = 0
        total_log_loss = 0
        total_cov_loss = 0
        step = 0
        # dataset[0] = encoder_batch_data dataset[1] = decoder_batch_data
        for encoder_batch_data, decoder_batch_data in dataset:
            # 获取相应的loss
            batch_loss, log_loss, cov_loss = train_step(encoder_batch_data["enc_input"],
                                                        encoder_batch_data["extended_enc_input"],
                                                        encoder_batch_data["max_oov_len"],
                                                        decoder_batch_data["dec_input"],
                                                        decoder_batch_data["dec_target"],
                                                        enc_pad_mask=encoder_batch_data["encoder_pad_mask"],
                                                        dec_pad_mask=decoder_batch_data["decoder_pad_mask"])
            total_loss += batch_loss
            total_log_loss += log_loss
            total_cov_loss += cov_loss
            step += 1
            # 每一步打印loss, 注意这里是总loss的平均值
            if step % 1 == 0:
                if params['use_coverage']:
                    print('Epoch {} Batch {} batch_loss {:.4f} avg_loss {:.4f} log_loss {:.4f} cov_loss {:.4f}'
                          .format(epoch + 1, step, batch_loss.numpy(), total_loss / step, total_log_loss / step,
                                  total_cov_loss / step))
                else:
                    print('Epoch {} Batch {} avg_loss {:.4f}'.format(epoch + 1, step, total_loss / step))
        # saving (checkpoint) the model every 2 epochs
        if (epoch + 1) % 1 == 0:
            if total_loss / step < best_loss:
                best_loss = total_loss / step
                ckpt_save_path = checkpoint_manager.save()
                print('Saving checkpoint for epoch {} at {} ,best loss {}'.format(epoch + 1, ckpt_save_path, best_loss))
                print('Epoch {} Loss {:.4f}'.format(epoch + 1, total_loss / step))
                print('Time taken for 1 epoch {} sec\n'.format(time.time() - start))
                # ---------------学习率衰减---------------------------------------------
                # lr = params["learning_rate"] * np.power(0.95, epoch + 1)
                # # 更新优化器的学习率
                # optimizer = tf.keras.optimizers.Adagrad(lr,
                #                                         initial_accumulator_value=params['adagrad_init_acc'],
                #                                         clipnorm=params['max_grad_norm'],
                #                                         epsilon=params['eps'])
                # assert lr == optimizer.get_config()["learning_rate"]
                # print("learning_rate=", optimizer.get_config()["learning_rate"])
                # ---------------------------------------------------------------------
