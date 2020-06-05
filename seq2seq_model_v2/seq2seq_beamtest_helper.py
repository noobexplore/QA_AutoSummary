#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2020/3/7 21:56
# @Author : TheTAO
# @Site : 
# @File : seq2seq_beamtest_helper.py
# @Software: PyCharm
import math
from tqdm import tqdm
import tensorflow as tf

"""
Hypothesis在整个beam运算的过程中用于结果保存的假设
"""


class Hypothesis:
    def __init__(self, tokens, log_probs, hidden, attn_dists):
        self.tokens = tokens  # 0-t时间步过程中产生的所有token的列表
        self.log_probs = log_probs  # 整个tokens列表的所对应的概率
        self.hidden = hidden  # 对应的最后一token的隐藏层状态
        self.attn_dists = attn_dists  # 对于所有token所对应的attention
        self.abstract = ""  # 生成的摘要

    def extend(self, token, log_prob, hidden, attn_dist):
        """
        利用解码出来得到到token来扩展当前的假设
        :param token: 解码得到的token
        :param log_prob: 对应的概率
        :param hidden: 对应的隐藏层状态
        :param attn_dist: 对应的attention权重
        :return: 新的假设
        """
        return Hypothesis(tokens=self.tokens + [token],  # we add the decoded token
                          log_probs=self.log_probs + [log_prob],  # we add the log prob of the decoded token
                          hidden=hidden,  # we update the state
                          attn_dists=self.attn_dists + [attn_dist])

    @property
    def latest_token(self):
        return self.tokens[-1]

    @property
    def tot_log_prob(self):
        return sum(self.log_probs)

    @property
    def avg_log_prob(self):
        return self.tot_log_prob / len(self.tokens)


# 打印出前k个预测结果
def print_top_k(hyp, k, vocab, batch):
    # 合并text
    text = " ".join([vocab.id_to_word(int(index)) for index in batch[0]])
    print('\nhyp.text :{}'.format(text))
    # 循环去取结果
    for i in range(min(k, len(hyp))):
        k_hyp = hyp[i]
        k_hyp.abstract = " ".join([vocab.id_to_word(index) for index in k_hyp.tokens])
        print('top {} best_hyp.abstract :{}\n'.format(i, k_hyp.abstract))


# beam搜索的过程
def beam_decode(model, batch, vocab, params):
    # 初始化mask
    batch_size = params['batch_size']
    start_index = vocab.word_to_id(vocab.START_DECODING)
    stop_index = vocab.word_to_id(vocab.STOP_DECODING)
    unk_index = vocab.word_to_id(vocab.UNKNOWN_TOKEN)

    # 单步的解码步骤，为的是获取每步的解码结果
    def decoder_one_step(enc_output, dec_input, dec_hidden):
        # 单个时间步 运行
        final_pred, dec_hidden, context_vector, attention_weights = model.call_decoder_onestep(dec_input, dec_hidden,
                                                                                               enc_output)
        # 这里获取到topk个token的概率以及ids，并且这里为了扩大搜索宽度一次罗列出了beamsize*2的k
        top_k_probs, top_k_ids = tf.nn.top_k(tf.squeeze(final_pred), k=params["beam_size"] * 2)
        # 计算每个topk的概率
        top_k_log_probs = tf.math.log(top_k_probs)
        # 保存产生的结果
        results = {
            "dec_hidden": dec_hidden,
            "attention_weights": attention_weights,
            "top_k_ids": top_k_ids,
            "top_k_log_probs": top_k_log_probs}
        # 返回需要保存的中间结果和概率
        return results

    # 测试数据的输入
    enc_input = batch
    # 根据上面的输入去计算相应的编码层的输出以及隐藏层的状态
    enc_output, enc_hidden = model.call_encoder(enc_input)
    # 初始化batch size个 假设对象
    hyps = [Hypothesis(tokens=[start_index], log_probs=[0.0], hidden=enc_hidden[0],
                       attn_dists=[]) for _ in range(batch_size)]
    # 初始化结果集
    results = []  # list to hold the top beam_size hypothesises
    # 遍历步数
    steps = 0  # initial step
    # 开始循环进行beamsearch，终止为长度不够或者结果集没达标
    while steps < params['max_dec_len'] and len(results) < params['beam_size']:
        # 获取最新待使用的token
        latest_tokens = [h.latest_token for h in hyps]
        # 替换掉oov的词
        latest_tokens = [t if t in vocab.id2word else unk_index for t in latest_tokens]
        # 获取所有的隐藏层状态
        hiddens = [h.hidden for h in hyps]
        # 获取解码层的输入与隐藏层状态
        dec_input = tf.expand_dims(latest_tokens, axis=1)
        dec_hidden = tf.stack(hiddens, axis=0)
        # 获取单步运行得到结果
        decoder_results = decoder_one_step(enc_output, dec_input, dec_hidden)
        # 解析并保存获取的结果
        dec_hidden = decoder_results['dec_hidden']
        attention_weights = decoder_results['attention_weights']
        top_k_log_probs = decoder_results['top_k_log_probs']
        top_k_ids = decoder_results['top_k_ids']
        # 现阶段全部可能的情况
        all_hyps = []
        # 原有的可能情况数量，这里特别的如果为起始输入的话因为是一样的所以就用一个初始假设
        num_orig_hyps = 1 if steps == 0 else len(hyps)
        # 遍历添加所有可能结果
        for i in range(num_orig_hyps):
            h, new_hidden, attn_dist = hyps[i], dec_hidden[i], attention_weights[i]
            # 分裂添加beam size种可能性，这里为了扩大搜索的宽度*2
            for j in range(params['beam_size'] * 2):
                # 构造可能的情况
                new_hyp = h.extend(token=top_k_ids[i, j].numpy(), log_prob=top_k_log_probs[i, j],
                                   hidden=new_hidden, attn_dist=attn_dist)
                # 添加可能情况
                all_hyps.append(new_hyp)
        # 重置新的假设
        hyps = []
        # 按照概率的大小来排序
        sorted_hyps = sorted(all_hyps, key=lambda h: h.avg_log_prob, reverse=True)
        # 从有序排列的假设中来选取最佳的前k个结果集
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
        # 步数增加
        steps += 1
    # 如果没有预测结果则将初始的假设赋值给结果集
    if len(results) == 0:
        results = hyps
    # 按照平均概率去排列结果集
    hyps_sorted = sorted(results, key=lambda h: h.avg_log_prob, reverse=True)
    # 打印出最佳前3个结果
    # print_top_k(hyps_sorted, 3, vocab, batch)
    # 去取得最佳的结果集
    best_hyp = hyps_sorted[0]
    best_hyp.abstract = "".join(
        [vocab.id_to_word(index) for index in best_hyp.tokens if index != start_index and index != stop_index])
    return best_hyp


# 进度条的方式去存储beamSearch的结果
def beam_tpqm_decode(model, test_X, batch_size, vocab, params):
    # 存储结果
    results = []
    # 样本数量
    sample_size = len(test_X)
    # batch 操作轮数 math.ceil向上取整 小数 +1，因为最后一个batch可能不足一个batch size 大小 ,但是依然需要计算
    steps_epoch = math.ceil(sample_size / batch_size)
    for i in tqdm(range(steps_epoch)):
        batch_data = test_X[i * batch_size:(i + 1) * batch_size]
        # 这里转为tensor
        beam_search_data = beam_generator(batch_data, batch_size)
        for b in beam_search_data:
            best_hyp = beam_decode(model, b, vocab, params)
            # print(best_hyp.abstract)
            results.append(best_hyp.abstract)
    return results


# 用于beamsearch的批量生成器
def beam_generator(test_X, beam_size):
    for row in test_X:
        beam_search_data = tf.convert_to_tensor([row for i in range(beam_size)])
        yield beam_search_data