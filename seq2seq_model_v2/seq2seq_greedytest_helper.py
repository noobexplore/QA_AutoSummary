#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2020/3/7 21:51
# @Author : TheTAO
# @Site : 
# @File : seq2seq_greedytest_helper.py
# @Software: PyCharm
import tensorflow as tf
import math
from tqdm import tqdm


# 批量的贪婪测试函数
def greedy_decode(model, data_X, batch_size, vocab, params):
    # 存储结果
    results = []
    # 样本数量
    sample_size = len(data_X)
    # batch 操作轮数 math.ceil向上取整 小数 +1，因为最后一个batch可能不足一个batch size 大小 ,但是依然需要计算
    steps_epoch = math.ceil(sample_size / batch_size)
    for i in tqdm(range(steps_epoch)):
        batch_data = data_X[i * batch_size:(i + 1) * batch_size]
        results += batch_greedy_decode(model, batch_data, vocab, params)
    return results


# 单个的贪婪测试函数
def batch_greedy_decode(model, batch_data, vocab, params):
    # 判断输入长度
    batch_size = len(batch_data)
    # 开辟结果存储list
    predicts = [''] * batch_size
    # 转化为tensor
    inps = tf.convert_to_tensor(batch_data)
    # 0. 初始化隐藏层输入
    hidden = [tf.zeros((batch_size, params['enc_units']))]
    # 1. 构建encoder
    enc_output, enc_hidden = model.encoder(inps, hidden)
    # 2. 复制
    dec_hidden = enc_hidden
    # 3.构造起始标志
    dec_input = tf.expand_dims([vocab.word_to_id(vocab.START_DECODING)] * batch_size, 1)
    # 计算初始的ct
    context_vector, _ = model.attention(dec_hidden, enc_output)
    # 步步开始解码预测
    for t in range(params['max_dec_len']):
        # 计算上下文和注意力权重
        context_vector, attention_weights = model.attention(dec_hidden, enc_output)
        # 单步解析预测
        predictions, dec_hidden = model.decoder(dec_input, dec_hidden, enc_output, context_vector)
        # id转换 贪婪搜索取最大概率结果
        predicted_ids = tf.argmax(predictions, axis=1).numpy()
        # 循环合并预测结果
        for index, predicted_id in enumerate(predicted_ids):
            predicts[index] += vocab.id_to_word(predicted_id) + ' '
        # 进行下一步预测
        dec_input = tf.expand_dims(predicted_ids, 1)
    # 预测结果
    results = []
    # 循环添加预测结果
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