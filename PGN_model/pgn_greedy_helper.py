#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2020/3/18 13:04
# @Author : TheTAO
# @Site : 
# @File : pgn_greedy_helper.py
# @Software: PyCharm
import tensorflow as tf
from tqdm import tqdm
from PGN_model.pgn_model import _calc_final_dist


def greedy_decode(model, dataset, vocab, params):
    # 存储结果
    batch_size = params["batch_size"]
    results = []
    sample_size = 20000
    # batch 操作轮数 math.ceil向上取整 小数 +1
    # 因为最后一个batch可能不足一个batch size 大小 ,但是依然需要计算
    steps_epoch = sample_size // batch_size + 1
    # [0,steps_epoch)
    ds = iter(dataset)
    for i in tqdm(range(steps_epoch)):
        enc_data, dec_data = next(ds)
        results += batch_greedy_decode(model, enc_data, vocab, params)
    return results


def batch_greedy_decode(model, encoder_batch_data, vocab, params):
    # 判断输入长度
    batch_data = encoder_batch_data["enc_input"]
    batch_size = encoder_batch_data["enc_input"].shape[0]
    # 开辟结果存储list
    predicts = [''] * batch_size
    inputs = batch_data
    # print(batch_size, batch_data.shape)
    enc_output, enc_hidden = model.encoder(inputs)
    dec_hidden = enc_hidden
    # dec_input = tf.expand_dims([vocab.word_to_id(vocab.START_DECODING)] * batch_size, 1)
    dec_input = tf.constant([vocab.word2id[vocab.START_DECODING]] * batch_size)
    # Teacher forcing - feeding the target as the next input
    try:
        batch_oov_len = tf.shape(encoder_batch_data["article_oovs"])[1]
    except:
        batch_oov_len = tf.constant(0)
    # 初始化coverage
    coverage = tf.zeros((enc_output.shape[0], enc_output.shape[1], 1))
    for t in range(params['max_dec_len']):
        # 单步预测
        # final_dist (batch_size, 1, vocab_size+batch_oov_len)
        final_dist, dec_hidden, coverage = model.call_decoder_one_step(dec_input,
                                                                       dec_hidden,
                                                                       enc_output,
                                                                       encoder_batch_data["extended_enc_input"],
                                                                       batch_oov_len,
                                                                       encoder_batch_data["encoder_pad_mask"],
                                                                       use_coverage=True,
                                                                       prev_coverage=coverage)
        # id转换
        final_dist = tf.squeeze(final_dist, axis=1)
        predicted_ids = tf.argmax(final_dist, axis=1)
        for index, predicted_id in enumerate(predicted_ids.numpy()):
            predicts[index] += vocab.id_to_word(predicted_id) + ' '
        # using teacher forcing
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
