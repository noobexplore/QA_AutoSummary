#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/5/28 11:28
# @Author  : TheTao
# @Site    : 
# @File    : model_layers.py
# @Software: PyCharm
import tensorflow as tf


class Encoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, embedding_matrix, enc_units, batch_size):
        super(Encoder, self).__init__()
        self.batch_size = batch_size
        self.enc_units = enc_units
        self.use_bi_gru = True
        # 如果采用双向gru则units需要对半
        if self.use_bi_gru:
            self.enc_units = self.enc_units // 2
        # embedding层
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim, weights=[embedding_matrix],
                                                   trainable=False)
        # GRU层
        self.gru = tf.keras.layers.GRU(self.enc_units,
                                       return_sequences=True,
                                       return_state=True,
                                       recurrent_initializer='glorot_uniform')
        # 双向GRU层
        self.bi_gru = tf.keras.layers.Bidirectional(self.gru)

    def call(self, enc_input):
        # (batch_size, enc_len, embedding_dim)
        enc_input_embedded = self.embedding(enc_input)
        # 初始化GRU层
        [initial_state] = self.gru.get_initial_state(enc_input_embedded)
        # 是否使用双向GRU
        if self.use_bi_gru:
            output, forward_state, backward_state = self.bi_gru(enc_input_embedded, initial_state=[initial_state] * 2)
            enc_hidden = tf.keras.layers.concatenate([forward_state, backward_state], axis=-1)
        else:
            output, enc_hidden = self.gru(enc_input_embedded, initial_state=[initial_state])
        return output, enc_hidden


class BahdanauAttention(tf.keras.layers.Layer):
    def __init__(self, units):
        super(BahdanauAttention, self).__init__()
        self.W_s = tf.keras.layers.Dense(units)
        self.W_h = tf.keras.layers.Dense(units)
        self.W_c = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, dec_hidden, enc_output, enc_pad_mask, prev_coverage, use_coverage=True):
        # dec_hidden shape == (batch_size, hidden size)
        # enc_output (batch_size, enc_len, enc_units)
        # hidden_with_time_axis shape == (batch_size, 1, dec_units)
        # we are doing this to perform addition to calculate the score
        hidden_with_time_axis = tf.expand_dims(dec_hidden, 1)
        if use_coverage:
            # Multiply coverage vector by w_c to get coverage_features.
            # self.W_s(values) [batch_sz, max_len, units] self.W_h(hidden_with_time_axis) [batch_sz, 1, units]
            # self.W_c(prev_coverage) [batch_sz, max_len, units]  score [batch_sz, max_len, 1]
            score = self.V(tf.nn.tanh(self.W_s(enc_output) + self.W_h(hidden_with_time_axis) + self.W_c(prev_coverage)))
            # attention_weights shape (batch_size, max_len, 1)
            # attention_weights shape (batch_size, max_length, 1)
            attention_weights = tf.nn.softmax(score, axis=1)
            # attention_weights = masked_attention(enc_pad_mask, attention_weights)
            coverage = attention_weights + prev_coverage
            # 使用注意力权重*编码器输出作为返回值，将来会作为解码器的输入
        # context_vector shape after sum == (batch_size, enc_units)
        context_vector = attention_weights * enc_output
        context_vector = tf.reduce_sum(context_vector, axis=1)
        return context_vector, tf.squeeze(attention_weights, -1), coverage


# 解码部分是改进重点
class Decoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, embedding_matrix, dec_units, batch_size, attention):
        super(Decoder, self).__init__()
        self.batch_sz = batch_size
        self.dec_units = dec_units
        # 词向量层
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim, weights=[embedding_matrix],
                                                   trainable=False)
        # GRU
        self.cell = tf.keras.layers.GRUCell(units=self.dec_units,
                                            recurrent_initializer='glorot_uniform')
        # 注意力
        self.attention = attention
        # 加入两层的全连接层
        self.fc1 = tf.keras.layers.Dense(self.dec_units * 2)
        self.fc2 = tf.keras.layers.Dense(vocab_size)

    def call(self, dec_input,  # (batch_size, )
             prev_dec_hidden,  # (batch_size, dec_units)
             enc_output,  # (batch_size, enc_len, enc_units)
             enc_pad_mask,  # (batch_size, enc_len)
             prev_coverage,  # (batch_size, enc_len, 1)
             use_coverage=True):
        # 得到词向量, output[2]
        # dec_x (batch_size, embedding_dim)
        dec_x = self.embedding(dec_input)
        # 此处为beam_search的时候才要加这句代码
        dec_x = tf.squeeze(dec_x, 1)
        # 应用GRU单元算出dec_hidden
        # 注意cell 返回的state是一个列表，gru单元中为 [h] lstm [h, c]
        # 所以这里用[dec_hidden] 取出来，这样dec_hidden就是tensor形式了
        # dec_output (batch_size, dec_units)
        # dec_hidden (batch_size, dec_units), output[1]

        dec_output, [dec_hidden] = self.cell(dec_x, [prev_dec_hidden])
        # 计算注意力，得到上下文，注意力分布，coverage
        # context_vector (batch_size, enc_units), output[0]
        # attn (batch_size, enc_len), output[4]
        # coverage (batch_size, enc_len, 1), output[5]
        context_vector, attn, coverage = self.attention(dec_hidden, enc_output, enc_pad_mask, prev_coverage,
                                                        use_coverage)
        # 将上一循环的预测结果跟注意力权重值结合在一起来预测vocab的分布
        # dec_output (batch_size, enc_units + dec_units)
        dec_output = tf.concat([dec_output, context_vector], axis=-1)
        # pred (batch_size, enc_units + dec_units)
        pred = self.fc1(dec_output)
        # pred (batch_size, vocab), output[3]
        pred = self.fc2(pred)
        """output
        output[0]: context_vector (batch_size, dec_units)
        output[1]: dec_hidden (batch_size, dec_units)
        output[2]: dec_x (batch_size, embedding_dim)
        output[3]: pred (batch_size, vocab_size)
        output[4]: attn (batch_size, enc_len)
        output[5]: coverage (batch_size, enc_len, 1)
        """
        return context_vector, dec_hidden, dec_x, pred, attn, coverage


class Pointer(tf.keras.layers.Layer):
    def __init__(self):
        super(Pointer, self).__init__()
        self.w_s_reduce = tf.keras.layers.Dense(1)
        self.w_i_reduce = tf.keras.layers.Dense(1)
        self.w_c_reduce = tf.keras.layers.Dense(1)

    def call(self, context_vector, dec_hidden, dec_inp):
        return tf.nn.sigmoid(self.w_s_reduce(dec_hidden) + self.w_c_reduce(context_vector) + self.w_i_reduce(dec_inp))
