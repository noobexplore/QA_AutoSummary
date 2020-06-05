#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/5/30 9:24
# @Author  : TheTao
# @Site    : 
# @File    : pgn_test_helper.py
# @Software: PyCharm
from tqdm import tqdm
import tensorflow as tf
from PGN_remodel.PGN_model import _calc_final_dist


def greedy_decode(model, dataset, vocab, params):
    # 存储结果
    results = []
    batch_size = params["batch_size"]
    sample_size = 20000
    # 这里说实话最好整除，因为next取出的为一个batch长度感觉是循环取出的
    steps_epoch = sample_size // batch_size
    ds = iter(dataset)
    for _ in tqdm(range(steps_epoch)):
        enc_data, dec_data = next(ds)
        results += batch_greedy_decode(model, enc_data, vocab, params)
    return results


def decode_one_step(params, model, enc_extended_inp, batch_oov_len, dec_input, dec_hidden, enc_output,
                    enc_pad_mask, prev_coverage, batch_size, use_coverage=True):
    # 开始decoder
    context_vector, dec_hidden, dec_x, pred, attn, coverage \
        = model.decoder(dec_input, dec_hidden, enc_output, enc_pad_mask, prev_coverage, use_coverage)
    # 计算p_gen
    p_gen = model.pointer(context_vector, dec_hidden, dec_x)
    # 保证pred attn p_gen的参数为3D的
    final_dist = _calc_final_dist(enc_extended_inp,
                                  tf.expand_dims(pred, 1),
                                  tf.expand_dims(attn, 1),
                                  tf.expand_dims(p_gen, 1),
                                  batch_oov_len,
                                  params["vocab_size"],
                                  batch_size)
    return final_dist, dec_hidden, coverage


def batch_greedy_decode(model, enc_data, vocab, params):
    # 判断输入长度
    batch_data = enc_data["enc_input"]
    batch_size = enc_data["enc_input"].shape[0]
    # 开辟结果存储list
    predicts = [''] * batch_size
    inputs = batch_data
    enc_output, enc_hidden = model.encoder(inputs)
    dec_hidden = enc_hidden
    dec_input = tf.constant([vocab.word2id[vocab.START_DECODING]] * batch_size)
    try:
        batch_oov_len = tf.shape(enc_data["article_oovs"])[1]
    except:
        batch_oov_len = tf.constant(0)

    coverage = tf.zeros((enc_output.shape[0], enc_output.shape[1], 1))
    for t in range(params['max_dec_len']):
        # 单步预测
        final_dist, dec_hidden, coverage = decode_one_step(params, model,
                                                           enc_data["extended_enc_input"],
                                                           batch_oov_len,
                                                           dec_input,
                                                           dec_hidden,
                                                           enc_output,
                                                           enc_data["enc_mask"],
                                                           coverage,
                                                           batch_size,
                                                           use_coverage=True)
        # id转换
        final_dist = tf.squeeze(final_dist, axis=1)
        predicted_ids = tf.argmax(final_dist, axis=1)
        for index, predicted_id in enumerate(predicted_ids.numpy()):
            predicts[index] += vocab.id_to_word(predicted_id) + ' '
        dec_input = predicted_ids

    results = []
    for predict in predicts:
        # 去掉句子前后空格
        predict = predict.strip()
        # 句子小于max len就结束了 截断
        if vocab.STOP_DECODING in predict:
            # 截断stop
            predict = predict[:predict.index(vocab.STOP_DECODING)]
        # 保存结果
        results.append(predict)
    return results
