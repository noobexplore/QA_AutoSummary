#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2020/3/7 17:08
# @Author : TheTAO
# @Site : 
# @File : seq2seq_model.py
# @Software: PyCharm
import tensorflow as tf
from seq2seq_model_v2.model_layer import Encoder, BahdanauAttention, Decoder
from seq2seq_model_v2.data_utils.wv_loader import load_embedding_matrix


class Seq2Seq(tf.keras.Model):
    def __init__(self, params):
        super(Seq2Seq, self).__init__()
        self.embedding_matrix = load_embedding_matrix()
        self.params = params
        # 编码层
        self.encoder = Encoder(params["vocab_size"], params["embed_size"],
                               self.embedding_matrix, params["enc_units"],
                               params["batch_size"])
        # 注意力层
        self.attention = BahdanauAttention(params["attn_units"])
        # 解码层
        self.decoder = Decoder(params["vocab_size"], params["embed_size"],
                               self.embedding_matrix, params["dec_units"],
                               params["batch_size"])

    # 编码层的编码操作
    def call_encoder(self, enc_inp):
        enc_hidden = self.encoder.initialize_hidden_state()
        enc_output, enc_hidden = self.encoder(enc_inp, enc_hidden)
        return enc_output, enc_hidden

    # 单步的解码操作
    def call_decoder_onestep(self, dec_input, dec_hidden, enc_output):
        # 计算注意力与上下文向量
        context_vector, attention_weights = self.attention(dec_hidden, enc_output)
        # 解码操作
        pred, dec_hidden = self.decoder(dec_input, None, None, context_vector)
        return pred, dec_hidden, context_vector, attention_weights

    # 解码操作
    def call(self, dec_input, dec_hidden, enc_output, dec_target):
        predictions = []
        attentions = []
        context_vector, _ = self.attention(dec_hidden, enc_output)
        for t in range(1, dec_target.shape[1]):
            pred, dec_hidden = self.decoder(dec_input, dec_hidden,
                                            enc_output, context_vector)
            context_vector, attn = self.attention(dec_hidden, enc_output)
            # using teacher forcing
            dec_input = tf.expand_dims(dec_target[:, t], 1)
            predictions.append(pred)
            attentions.append(attn)
        return tf.stack(predictions, 1), dec_hidden
