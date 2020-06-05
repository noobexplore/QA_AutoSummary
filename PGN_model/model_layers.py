#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2020/3/9 15:07
# @Author : TheTAO
# @Site : 
# @File : model_layers.py
# @Software: PyCharm
import tensorflow as tf
from PGN_model.data_utils.params_utils import get_params
from PGN_model.data_utils.config import vocab_path
from PGN_model.data_utils.gpu_utils import config_gpu
from PGN_model.data_utils.wv_loader import load_embedding_matrix, Vocab

params = get_params()


class Encoder(tf.keras.Model):
    def __init__(self, embedding_matrix, enc_units, batch_size):
        super(Encoder, self).__init__()
        self.batch_size = batch_size
        self.enc_units = enc_units // 2
        self.vocab_size, self.embedding_dim = embedding_matrix.shape
        self.embedding = tf.keras.layers.Embedding(self.vocab_size, self.embedding_dim, weights=[embedding_matrix],
                                                   trainable=False)
        # 初始化gru层
        self.gru = tf.keras.layers.GRU(self.enc_units,
                                       return_sequences=True,
                                       return_state=True,
                                       recurrent_initializer='glorot_uniform')
        # 连接两个gru层
        self.bi_gru = tf.keras.layers.Bidirectional(self.gru)

    def __call__(self, x):
        x = self.embedding(x)
        [initial_state] = self.gru.get_initial_state(x)
        if params['bi_gru']:
            output, forward_state, backward_state = self.bi_gru(x, initial_state=[initial_state] * 2)
            hidden = tf.keras.layers.concatenate([forward_state, backward_state], axis=-1)
        else:
            output, hidden = self.gru(x, initial_state=[initial_state])
        return output, hidden


class BahdanauAttention(tf.keras.layers.Layer):
    def __init__(self, units):
        super(BahdanauAttention, self).__init__()
        self.W_s = tf.keras.layers.Dense(units)
        self.W_h = tf.keras.layers.Dense(units)
        self.W_c = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def __call__(self, dec_hidden, enc_output, enc_pad_mask, use_coverage=True, prev_coverage=None):
        # hidden shape == (batch_size, hidden size)
        # hidden_with_time_axis shape == (batch_size, 1, hidden size)
        # we are doing this to perform addition to calculate the score
        hidden_with_time_axis = tf.expand_dims(dec_hidden, 1)

        def masked_attention(score):
            """Take softmax of e then apply enc_padding_mask and re-normalize"""
            attn_dist = tf.squeeze(score, axis=2)  # shape=(16, 200)
            attn_dist = tf.nn.softmax(attn_dist, axis=1)  # shape=(16, 200)
            mask = tf.cast(enc_pad_mask, dtype=attn_dist.dtype)
            attn_dist *= mask
            masked_sums = tf.reduce_sum(attn_dist, axis=1)
            attn_dist = attn_dist / tf.reshape(masked_sums, [-1, 1])
            attn_dist = tf.expand_dims(attn_dist, axis=2)
            return attn_dist

        if use_coverage and prev_coverage is not None:
            score = self.V(tf.nn.tanh(self.W_s(enc_output) + self.W_h(hidden_with_time_axis) + self.W_c(prev_coverage)))
            # attention_weights shape (batch_size, max_len, 1)
            attn_dist = masked_attention(score)
            # attention_weights sha== (batch_size, max_length, 1)
            coverage = attn_dist + prev_coverage
        else:
            # 计算注意力权重值
            score = self.V(tf.nn.tanh(self.W_s(enc_output) + self.W_h(hidden_with_time_axis)))
            attn_dist = masked_attention(score)
            if use_coverage:  # first step of training
                coverage = attn_dist  # initialize coverage
            else:
                coverage = []
        # # 使用注意力权重*编码器输出作为返回值，将来会作为解码器的输入
        # context_vector shape after sum == (batch_size, hidden_size)
        context_vector = attn_dist * enc_output
        context_vector = tf.reduce_sum(context_vector, axis=1)
        # tf.squeeze删除axis的维度的值
        # tf.squeeze(attn_dist, -1) == tf.squeeze(attn_dist, 2) (batch_size, sequence_length, 1)将这里的1删除
        return context_vector, tf.squeeze(attn_dist, -1), coverage


class Decoder(tf.keras.Model):
    def __init__(self, embedding_matrix, dec_units, batch_sz):
        super(Decoder, self).__init__()
        self.batch_sz = batch_sz
        self.dec_units = dec_units
        self.vocab_size, self.embedding_dim = embedding_matrix.shape
        self.embedding = tf.keras.layers.Embedding(self.vocab_size, self.embedding_dim, weights=[embedding_matrix],
                                                   trainable=False)
        self.gru = tf.keras.layers.GRU(self.dec_units,
                                       return_sequences=True,
                                       return_state=True,
                                       recurrent_initializer='glorot_uniform')
        self.fc = tf.keras.layers.Dense(self.vocab_size, activation=tf.keras.activations.softmax)

    def __call__(self, dec_inp, hidden, enc_output, context_vector):
        # 使用上次的隐藏层（第一次使用编码器隐藏层）、编码器输出计算注意力权重
        # enc_output shape == (batch_size, max_length, hidden_size)
        # x shape after passing through embedding == (batch_size, 1, embedding_dim)
        # print('x:{}'.format(x))
        dec_inp = self.embedding(dec_inp)
        # 将上一循环的预测结果跟注意力权重值结合在一起作为本次的GRU网络输入
        # x shape after concatenation == (batch_size, 1, embedding_dim + hidden_size)
        dec_x = tf.concat([tf.expand_dims(context_vector, 1), dec_inp], axis=-1)
        # passing the concatenated vector to the GRU
        output, state = self.gru(dec_x)
        # output shape == (batch_size * 1, hidden_size)
        output = tf.reshape(output, (-1, output.shape[2]))
        # output shape == (batch_size, vocab)
        prediction = self.fc(output)
        return dec_x, prediction, state


class Pointer(tf.keras.layers.Layer):
    def __init__(self):
        super(Pointer, self).__init__()
        self.w_s_reduce = tf.keras.layers.Dense(1)
        self.w_i_reduce = tf.keras.layers.Dense(1)
        self.w_c_reduce = tf.keras.layers.Dense(1)

    def __call__(self, context_vector, state, dec_inp):
        # 统一输入
        dec_inp = tf.squeeze(dec_inp, axis=1)
        return tf.nn.sigmoid(self.w_s_reduce(state) + self.w_c_reduce(context_vector) + self.w_i_reduce(dec_inp))


# 测试一下
if __name__ == '__main__':
    # GPU资源配置
    config_gpu()
    # 读取vocab训练
    vocab = Vocab(vocab_path)
    # 使用GenSim训练好的embedding matrix
    embedding_matrix = load_embedding_matrix()
    # 初始化参数
    enc_max_len = 200
    dec_max_len = 50
    batch_size = 32
    embedding_dim = 300
    units = 512
    # 编码器结构
    encoder = Encoder(embedding_matrix, units, batch_size)
    # encoder input
    enc_inp = tf.ones(shape=(batch_size, enc_max_len), dtype=tf.int32)
    # decoder input
    dec_inp = tf.ones(shape=(batch_size, dec_max_len), dtype=tf.int32)
    # enc pad mask
    enc_pad_mask = tf.ones(shape=(batch_size, enc_max_len), dtype=tf.int32)
    # encoder hidden
    # enc_hidden = encoder.initialize_hidden_state()
    enc_output, enc_hidden = encoder(enc_inp)
    # 打印结果
    print('Encoder output shape: (batch size, sequence length, units) {}'.format(enc_output.shape))
    print('Encoder Hidden state shape: (batch size, units) {}'.format(enc_hidden.shape))
    attention_layer = BahdanauAttention(10)
    context_vector, attention_weights, coverage = attention_layer(enc_hidden, enc_output, enc_pad_mask)
    print("Attention context_vector shape: (batch size, units) {}".format(context_vector.shape))
    print("Attention weights shape: (batch_size, sequence_length, 1) {}".format(attention_weights.shape))
    print("Attention coverage: (batch_size, ) {}".format(coverage.shape))
    decoder = Decoder(embedding_matrix, units, batch_size)
    dec_x, dec_out, dec_hidden, = decoder(tf.random.uniform((32, 1)), enc_hidden, enc_output, context_vector)
    print('Decoder output shape: (batch_size, vocab size) {}'.format(dec_out.shape))
    print('Decoder dec_x shape: (batch_size, 1,embedding_dim + units) {}'.format(dec_x.shape))
    pointer = Pointer()
    p_gen = pointer(context_vector, dec_hidden, dec_x)
    print('Pointer p_gen shape: (batch_size,1) {}'.format(p_gen.shape))
