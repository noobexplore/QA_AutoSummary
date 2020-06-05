#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2020/3/9 15:57
# @Author : TheTAO
# @Site : 
# @File : pgn_model.py
# @Software: PyCharm
import tensorflow as tf
import numpy as np
from PGN_model.data_utils.params_utils import get_params
from PGN_model.data_utils.config import vocab_path
from PGN_model.data_utils.gpu_utils import config_gpu
from PGN_model.data_utils.wv_loader import load_embedding_matrix, Vocab
from PGN_model.model_layers import Encoder, Decoder, Pointer, BahdanauAttention
from PGN_model.batcher import batcher
from PGN_model.loss import calc_loss, loss_function


class PGN(tf.keras.Model):
    def __init__(self, params):
        super(PGN, self).__init__()
        self.embedding_matrix = load_embedding_matrix()
        self.params = params
        self.encoder = Encoder(self.embedding_matrix,
                               params["enc_units"],
                               params["batch_size"])
        self.attention = BahdanauAttention(params["attn_units"])
        self.decoder = Decoder(self.embedding_matrix,
                               params["dec_units"],
                               params["batch_size"])
        self.pointer = Pointer()

    def call_encoder(self, enc_inp):
        enc_output, enc_hidden = self.encoder(enc_inp)
        return enc_output, enc_hidden

    def call_decoder_one_step(self, dec_input, dec_hidden, enc_output, enc_extended_inp, batch_oov_len, enc_pad_mask,
                              use_coverage, prev_coverage):
        """
        单步解码，后面测试使用
        """
        context_v, attn_dist, coverage_next = self.attention(dec_hidden, enc_output, enc_pad_mask, use_coverage,
                                                             prev_coverage)
        dec_x, pred_dist, dec_hidden = self.decoder(dec_input, dec_hidden, enc_output, context_v)
        if self.params["pointer_gen"]:
            p_gen = self.pointer(context_v, dec_hidden, dec_x)
            final_dists = _calc_final_dist(enc_extended_inp,
                                           [pred_dist],
                                           [attn_dist],
                                           [p_gen],
                                           batch_oov_len,
                                           self.params["vocab_size"],
                                           self.params["batch_size"])
            return tf.stack(final_dists, 1), dec_hidden, context_v, attn_dist, p_gen, coverage_next
        return pred_dist, dec_hidden, context_v, attn_dist, coverage_next

    def call(self, dec_hidden, enc_output, dec_input, enc_extended_inp, batch_oov_len, enc_pad_mask, use_coverage,
             prev_coverage=None):
        """
        训练时解码过程
        """
        # 需要保存的一些值
        predictions = []
        attentions = []
        p_gens = []
        coverages = []
        # 计算attention
        context_v, _, coverage_next = self.attention(dec_hidden, enc_output, enc_pad_mask, use_coverage,
                                                     prev_coverage)
        # 一步步解码使用teacher forcing策略
        for t in range(dec_input.shape[1]):
            dec_x, pred_dist, dec_hidden = self.decoder(tf.expand_dims(dec_input[:, t], 1), dec_hidden, enc_output,
                                                        context_v)
            context_vector, attn_dist, coverage_next = self.attention(dec_hidden, enc_output, enc_pad_mask,
                                                                      use_coverage,
                                                                      coverage_next)
            # 保存中间结果
            p_gen = self.pointer(context_vector, dec_hidden, dec_x)
            coverages.append(coverage_next)  # (dec_step, batch_size, enc_len, 1)
            attentions.append(attn_dist)  # (dec_step, batch_size, enc_len)
            predictions.append(pred_dist)
            p_gens.append(p_gen)
        if self.params["pointer_gen"]:
            final_dists = _calc_final_dist(enc_extended_inp,
                                           predictions,
                                           attentions,
                                           p_gens,
                                           batch_oov_len,
                                           self.params["vocab_size"],
                                           self.params["batch_size"])
            if self.params["mode"] == "train":
                coverages = tf.stack(coverages, 1)  # (batch_size, dec_step, enc_len, 1)
                coverages = tf.squeeze(coverages, -1)  # (batch_size, dec_step, enc_len)
                attentions = tf.stack(attentions, 1)  # (batch_size, dec_step, enc_len)
                # attentions.shape = coverages.shape
                return final_dists, dec_hidden, attentions, coverages
            else:
                return tf.stack(final_dists, 1), dec_hidden, attentions, tf.stack(coverages, 1)
        else:
            return tf.stack(predictions, 1), dec_hidden, attentions, None, None


def _calc_final_dist(_enc_batch_extend_vocab, vocab_dists, attn_dists, p_gens, batch_oov_len, vocab_size,
                     batch_size):
    """
        Calculate the final distribution, for the pointer-generator model
        Args:
        vocab_dists: The vocabulary distributions. List length max_dec_steps of (batch_size, vsize) arrays.
                    The words are in the order they appear in the vocabulary file.
        attn_dists: The attention distributions. List length max_dec_steps of (batch_size, attn_len) arrays
        Returns:
        final_dists: The final distributions. List length max_dec_steps of (batch_size, extended_vsize) arrays.
        """
    # Multiply vocab dists by p_gen and attention dists by (1-p_gen)
    vocab_dists = [p_gen * dist for (p_gen, dist) in zip(p_gens, vocab_dists)]
    attn_dists = [(1 - p_gen) * dist for (p_gen, dist) in zip(p_gens, attn_dists)]

    # Concatenate some zeros to each vocabulary dist, to hold the probabilities for in-article OOV words
    # the maximum (over the batch) size of the extended vocabulary
    extended_size = vocab_size + batch_oov_len
    extra_zeros = tf.zeros((batch_size, batch_oov_len))
    # list length max_dec_steps of shape (batch_size, extended_size)
    vocab_dists_extended = [tf.concat(axis=1, values=[dist, extra_zeros]) for dist in vocab_dists]

    # Project the values in the attention distributions onto the appropriate entries in the final distributions
    # This means that if a_i = 0.1 and the ith encoder word is w, and w has index 500 in the vocabulary
    # then we add 0.1 onto the 500th entry of the final distribution
    # This is done for each decoder timestep.
    # This is fiddly; we use tf.scatter_nd to do the projection
    batch_nums = tf.range(0, limit=batch_size)  # shape (batch_size)
    batch_nums = tf.expand_dims(batch_nums, 1)  # shape (batch_size, 1)

    attn_len = tf.shape(_enc_batch_extend_vocab)[1]  # number of states we attend over
    batch_nums = tf.tile(batch_nums, [1, attn_len])  # shape (batch_size, attn_len)
    indices = tf.stack((batch_nums, _enc_batch_extend_vocab), axis=2)  # shape (batch_size, enc_t, 2)
    shape = [batch_size, extended_size]
    # list length max_dec_steps (batch_size, extended_size)
    attn_dists_projected = [tf.scatter_nd(indices, copy_dist, shape) for copy_dist in attn_dists]

    # Add the vocab distributions and the copy distributions together to get the final distributions
    # final_dists is a list length max_dec_steps; each entry is a tensor shape (batch_size, extended_size) giving
    # the final distribution for that decoder timestep
    # Note that for decoder timesteps and examples corresponding to a [PAD] token, this is junk - ignore.
    final_dists = [vocab_dist + copy_dist for (vocab_dist, copy_dist) in
                   zip(vocab_dists_extended, attn_dists_projected)]

    return final_dists


# 测试1个batch能否计算
def test_one_batch_result(b, model, params):
    encoder_batch_data, decoder_batch_data = next(iter(b))
    enc_input = encoder_batch_data["enc_input"]
    print('enc_input: {}'.format(enc_input.shape))
    extended_enc_input = encoder_batch_data["extended_enc_input"]
    max_oov_len = encoder_batch_data["max_oov_len"]
    dec_input = decoder_batch_data["dec_input"]
    print('dec_input: {}'.format(dec_input))
    dec_target = decoder_batch_data["dec_target"]
    print('dec_target: {}'.format(dec_target))
    enc_pad_mask = encoder_batch_data["encoder_pad_mask"]
    dec_pad_mask = decoder_batch_data["decoder_pad_mask"]
    enc_output, enc_hidden = model.call_encoder(enc_input)
    dec_hidden = enc_hidden
    # 模型计算的值
    final_dists, dec_hidden, attentions, coverages = model(dec_hidden,
                                                           enc_output,
                                                           dec_input,
                                                           extended_enc_input,
                                                           max_oov_len,
                                                           enc_pad_mask=enc_pad_mask,
                                                           use_coverage=params['use_coverage'],
                                                           prev_coverage=None)
    # 打印一些值
    print("dec_target shape is:{}".format(dec_target.shape))  # (1, 50)
    print("dec_hidden shape:{}".format(dec_hidden.shape))  # (1, 256)
    print("final_dists shape is:{}".format(np.array(final_dists).shape))  # (50, 1, 31817)
    print("attentions shape:{}".format(np.array(attentions).shape))  # (1, 50, 199)
    print("coverages shape:{}".format(np.array(coverages).shape))  # (1, 50, 199)
    # 测试loss 这里加上coverages传入到用于cov_loss的计算与罗老师的不一样
    batch_loss, pgn_loss, cover_loss = calc_loss(dec_target, final_dists, dec_pad_mask, attentions, coverages,
                                                 params['cov_loss_wt'],
                                                 params['use_coverage'],
                                                 params['pointer_gen'])
    # 打印loss结果
    print("batch_loss is:{}".format(batch_loss))
    print("pgn_loss is:{}".format(pgn_loss))
    print("cover_loss is:{}".format(cover_loss))


# 测试一下
if __name__ == '__main__':
    # GPU资源配置
    config_gpu()
    params = get_params()
    # 读取vocab训练
    vocab = Vocab(vocab_path)
    # 使用GenSim训练好的embedding matrix
    embedding_matrix = load_embedding_matrix()
    # build model
    params['mode'] = 'train'
    params['max_enc_len'] = 200
    params['max_dec_len'] = 50
    params['batch_size'] = 1
    params['pointer_gen'] = True
    params['use_coverage'] = True
    params['max_vocab_size'] = 50000
    params['cov_loss_wt'] = 1.0
    batcher = batcher(vocab, params)
    model = PGN(params)
    # encoder input
    enc_inp = tf.ones(shape=(params["batch_size"], params["max_enc_len"]), dtype=tf.int32)
    # enc pad mask
    enc_pad_mask = tf.ones(shape=(params["batch_size"], params["max_enc_len"]), dtype=tf.int32)
    # decoder input
    dec_inp = tf.ones(shape=(params["batch_size"], params["max_dec_len"]), dtype=tf.int32)
    # 获取编码输出与编码隐藏层
    enc_output, enc_hidden = model.encoder(enc_inp)
    # 打印结果
    print('Encoder output shape: (batch size, sequence length, units) {}'.format(enc_output.shape))
    print('Encoder Hidden state shape: (batch size, units) {}'.format(enc_hidden.shape))
    # 计算context_vector、attention_weights和coverage
    context_vector, attention_weights, coverage = model.attention(enc_hidden, enc_output, enc_pad_mask)
    print("Attention context_vector shape: (batch size, units) {}".format(context_vector.shape))
    print("Attention weights shape: (batch_size, sequence_length, 1) {}".format(attention_weights.shape))
    print("Attention coverage: (batch_size, ) {}".format(coverage.shape))
    dec_x_concat, dec_out, dec_hidden, = model.decoder(tf.random.uniform((1, 1)), enc_hidden, enc_output,
                                                       context_vector)
    print('Decoder output shape: (batch_size, vocab size) {}'.format(dec_out.shape))
    print('Decoder dec_x shape: (batch_size, 1, embedding_dim + units) {}'.format(dec_x_concat.shape))
    p_gen = model.pointer(context_vector, dec_hidden, dec_x_concat)
    print('Pointer p_gen shape: (batch_size,1) {}'.format(p_gen.shape))
    test_one_batch_result(batcher, model, params)
