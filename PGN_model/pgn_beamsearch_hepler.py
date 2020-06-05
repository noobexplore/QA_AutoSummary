#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2020/3/13 10:29
# @Author : TheTAO
# @Site : 
# @File : pgn_beamsearch_hepler.py
# @Software: PyCharm
import tensorflow as tf
import numpy as np


class Hypothesis:
    """ Class designed to hold hypothesises throughout the beamSearch decoding """
    def __init__(self, tokens, log_probs, hidden, attn_dists, p_gens, coverage):
        # list of all the tokens from time 0 to the current time step t
        self.tokens = tokens
        # list of the log probabilities of the tokens of the tokens
        self.log_probs = log_probs
        # decoder hidden state after the last token decoding
        self.hidden = hidden
        # attention dists of all the tokens
        self.attn_dists = attn_dists
        # 保存每句话的p_gen值用于判断是从词典中拿词还是从OOV list中
        self.p_gens = p_gens
        # 记录的coverage的权重用于去惩罚重复的词
        self.coverage = coverage
        # 产生的摘要
        self.abstract = ""
        # 标签用于评估
        self.real_abstract = ""
        # 原文字
        self.article = ""

    # 用于更新假设的函数
    def extend(self, token, log_prob, hidden, attn_dist, p_gen, coverage):
        """
        Method to extend the current hypothesis by adding the next decoded token and
        all the informations associated with it
        """
        # 更新以上初始化的值
        return Hypothesis(tokens=self.tokens + [token],  # we add the decoded token
                          log_probs=self.log_probs + [log_prob],  # we add the log prob of the decoded token
                          hidden=hidden,  # we update the state
                          attn_dists=self.attn_dists + [attn_dist],
                          p_gens=self.p_gens + [p_gen],
                          coverage=coverage)

    @property
    def latest_token(self):
        return self.tokens[-1]

    @property
    def tot_log_prob(self):
        return sum(self.log_probs)

    @property
    def avg_log_prob(self):
        return self.tot_log_prob / len(self.tokens)


# 打印出前K个最佳的值
def print_top_k(hyp, k, vocab, batch):
    # 直接可以从batch中拿出原文
    text = batch[0]["article"].numpy()[0].decode()
    print('\nhyp.text :{}'.format(text))
    for i in range(min(k, len(hyp))):
        k_hyp = hyp[i]
        k_hyp = result_index2text(k_hyp, vocab, batch)
        print('top {} best_hyp.abstract :{}\n'.format(i, k_hyp.abstract))


# 将index转化为text
def result_index2text(hyp, vocab, batch):
    # batch[0] = encoder_batch_data batch[1] = decoder_batch_data
    article_oovs = batch[0]["article_oovs"].numpy()[0]
    hyp.real_abstract = batch[1]["abstract"].numpy()[0].decode()
    hyp.article = batch[0]["article"].numpy()[0].decode()
    words = []
    # 遍历解码产生的tokens去转化为词从而形成摘要
    for index in hyp.tokens:
        if index != vocab.START_DECODING_INDEX and index != vocab.STOP_DECODING_INDEX:
            if index < (len(article_oovs) + vocab.size()):
                if index < vocab.size():
                    words.append(vocab.id_to_word(index))
                else:
                    words.append(article_oovs[index - vocab.size()].decode())
            else:
                print('error values id :{}'.format(index))
    hyp.abstract = " ".join(words)
    return hyp


# beam解码
def beam_decode(model, batch, vocab, params):
    # 初始化mask
    start_index = vocab.word_to_id(vocab.START_DECODING)
    stop_index = vocab.word_to_id(vocab.STOP_DECODING)
    unk_index = vocab.word_to_id(vocab.UNKNOWN_TOKEN)
    batch_size = params['batch_size']

    # 单步的decoder以满足beamsearch的需要
    def decoder_one_step(enc_output, dec_input, dec_hidden, enc_extended_inp, batch_oov_len, enc_pad_mask,
                         use_coverage, prev_coverage):
        # 单个时间步骤去解码，并获取相应计算的值
        final_pred, dec_hidden, context_vector, \
        attention_weights, p_gens, converage_next = model.call_decoder_one_step(dec_input, dec_hidden, enc_output,
                                                                                enc_extended_inp, batch_oov_len,
                                                                                enc_pad_mask, use_coverage=use_coverage,
                                                                                prev_coverage=prev_coverage)

        # 拿到top k个index和概率
        top_k_probs, top_k_ids = tf.nn.top_k(tf.squeeze(final_pred), k=params["beam_size"] * 2)
        # 计算log概率
        top_k_log_probs = tf.math.log(top_k_probs)
        # 获取结果
        results = {"coverage_next": converage_next, "last_context_vector": context_vector, "dec_hidden": dec_hidden,
                   "attention_weights": attention_weights, "top_k_ids": top_k_ids, "top_k_log_probs": top_k_log_probs,
                   "p_gens": p_gens}
        # 返回一些中间结果
        return results

    # 开始解码
    # 首先获取测试数据的输入
    enc_input = batch[0]["enc_input"]
    enc_extended_inp = batch[0]["extended_enc_input"]
    batch_oov_len = batch[0]["max_oov_len"]
    enc_pad_mask = batch[0]["encoder_pad_mask"]
    # 计算编码层的输出
    enc_output, enc_hidden = model.call_encoder(enc_input)
    # 初始化batch size个假设对象
    hyps = [Hypothesis(tokens=[start_index],
                       log_probs=[0.0],
                       hidden=enc_hidden[0],
                       attn_dists=[],
                       p_gens=[],
                       # 初始化coverage
                       coverage=np.zeros([enc_input.shape[1], 1], dtype=np.float32)) for _ in range(batch_size)]
    # 初始化结果集
    results = []
    # 遍历步数
    steps = 0
    # 长度还不够并且结果还不够继续搜索
    while steps < params['max_dec_len'] and len(results) < params['beam_size']:
        # 获取最新待使用的token
        latest_tokens = [h.latest_token for h in hyps]
        # 替换掉 oov token->unknown token
        latest_tokens = [t if t in vocab.id2word else unk_index for t in latest_tokens]
        # 获取所以隐藏层状态
        hiddens = [h.hidden for h in hyps]
        # 获取先前的coverage
        prev_coverage = [h.coverage for h in hyps]
        prev_coverage = tf.convert_to_tensor(prev_coverage)
        # 构造解码的输入以及隐藏层状态
        dec_input = tf.expand_dims(latest_tokens, axis=1)
        dec_hidden = tf.stack(hiddens, axis=0)
        # 单步运行decoder计算需要的值
        decoder_results = decoder_one_step(enc_output, dec_input, dec_hidden, enc_extended_inp, batch_oov_len,
                                           enc_pad_mask,
                                           use_coverage=params['use_coverage'],
                                           prev_coverage=prev_coverage)
        # 解析获取到的结果
        dec_hidden = decoder_results['dec_hidden']
        attention_weights = decoder_results['attention_weights']
        top_k_log_probs = decoder_results['top_k_log_probs']
        top_k_ids = decoder_results['top_k_ids']
        new_coverage = decoder_results['coverage_next']
        p_gens = decoder_results['p_gens']
        # 现阶段全部可能情况
        all_hyps = []
        # 原有的可能情况数量
        num_orig_hyps = 1 if steps == 0 else len(hyps)
        # 遍历添加所有可能结果
        for i in range(num_orig_hyps):
            h, new_hidden, attn_dist = hyps[i], dec_hidden[i], attention_weights[i]
            if params['pointer_gen']:
                p_gen = p_gens[i]
            else:
                p_gen = 0
            if params['use_coverage']:
                new_coverage_i = new_coverage[i]
            else:
                new_coverage_i = 0
            # 分裂添加beam size种可能性
            for j in range(params['beam_size'] * 2):
                # 构造可能的情况
                new_hyp = h.extend(token=top_k_ids[i, j].numpy(),
                                   log_prob=top_k_log_probs[i, j],
                                   hidden=new_hidden,
                                   attn_dist=attn_dist,
                                   p_gen=p_gen,
                                   coverage=new_coverage_i)
                # 添加可能情况
                all_hyps.append(new_hyp)
        # 重置
        hyps = []
        # 按照概率来排序
        sorted_hyps = sorted(all_hyps, key=lambda h: h.avg_log_prob, reverse=True)
        # 筛选top前beam_size句话
        for h in sorted_hyps:
            if h.latest_token == stop_index:
                # 长度符合预期,遇到句尾,添加到结果集
                if steps >= params['min_dec_steps']:
                    results.append(h)
            else:
                # 未到结束 ,添加到假设集
                hyps.append(h)
            # 如果假设句子正好等于beam_size 或者结果集正好等于beam_size 就不在添加
            if len(hyps) == params['beam_size'] or len(results) == params['beam_size']:
                break
        steps += 1
    if len(results) == 0:
        results = hyps
    hyps_sorted = sorted(results, key=lambda h: h.avg_log_prob, reverse=True)
    # 打印出topk个结果
    # if params['mode'] == 'eval':
    #     print_top_k(hyps_sorted, params['beam_size'], vocab, batch)
    # 获取最佳的假设
    best_hyp = hyps_sorted[0]
    best_hyp = result_index2text(best_hyp, vocab, batch)
    if params['mode'] == 'eval':
        print('\nreal_article: {}'.format(best_hyp.real_abstract))
        print('article: {}'.format(best_hyp.abstract))
    return best_hyp
