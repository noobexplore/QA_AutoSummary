#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2020/3/7 21:52
# @Author : TheTAO
# @Site : 
# @File : seq2seq_greedytest.py
# @Software: PyCharm
import tensorflow as tf
import pandas as pd
from seq2seq_model_v2.seq2seq_beamtest_helper import beam_tpqm_decode
from seq2seq_model_v2.seq2seq_model import Seq2Seq
from seq2seq_model_v2.seq2seq_greedytest_helper import greedy_decode
from seq2seq_model_v2.data_utils.config import checkpoint_dir, test_data_path
from seq2seq_model_v2.data_utils.data_loader import load_test_dataset
from seq2seq_model_v2.data_utils.gpu_utils import config_gpu
from seq2seq_model_v2.data_utils.params_utils import get_params
from seq2seq_model_v2.data_utils.wv_loader import Vocab


# beam测试的入口函数
def test(params):
    assert params["mode"].lower() in ["test", "eval"], "change training mode to 'test' or 'eval'"
    # assert params["beam_size"] == params["batch_size"], "Beam size must be equal to batch_size, change the params"
    # GPU资源配置
    config_gpu()
    print("初始化模型...")
    model = Seq2Seq(params)
    print("加载词典...")
    vocab = Vocab(params["vocab_path"], params["vocab_size"])
    print("创建检查点管理器，重新加载模型...")
    checkpoint = tf.train.Checkpoint(Seq2Seq=model)
    checkpoint_manager = tf.train.CheckpointManager(checkpoint, checkpoint_dir, max_to_keep=5)
    checkpoint.restore(checkpoint_manager.latest_checkpoint)
    if checkpoint_manager.latest_checkpoint:
        print("Restored from {}".format(checkpoint_manager.latest_checkpoint))
    else:
        print("Initializing from scratch.")
    print("Model restored")
    if params['greedy_decode']:
        predict_result(model, params, vocab, params['result_save_path'])
    else:
        beam_result(model, params, vocab, params['result_save_path'])
        # b = beam_test_batch_generator(params["beam_size"])
        # results = []
        # for batch in b:
        #     best_hyp = beam_decode(model, batch, vocab, params)
        #     results.append(best_hyp.abstract)
        # save_predict_result(results, params['result_save_path'])
        # print('save result to :{}'.format(params['result_save_path']))


# beamSearch批量预测入口
def beam_result(model, params, vocab, result_save_path):
    test_X = load_test_dataset(params['max_enc_len'])
    # test_one = test_X[:4]
    # 预测结果
    results = beam_tpqm_decode(model, test_X, params["beam_size"], vocab, params)
    print('最后的预测结果的长度：{}'.format(len(results)))
    # 保存结果
    save_predict_result(results, result_save_path)


# 贪婪预测函数入口
def predict_result(model, params, vocab, result_save_path):
    test_X = load_test_dataset(params['max_enc_len'])
    # 预测结果
    results = greedy_decode(model, test_X, params['batch_size'], vocab, params)
    # 保存结果
    save_predict_result(results, result_save_path)


# 存储结果保存为CSV文件
def save_predict_result(results, result_save_path):
    # 读取结果
    test_df = pd.read_csv(test_data_path)
    # 填充结果
    test_df['Prediction'] = results
    # 提取ID和预测结果两列
    test_df = test_df[['QID', 'Prediction']]
    # 批量去掉空格和特殊字符
    test_df['Prediction'] = test_df['Prediction'].apply(submit_proc)
    # 保存结果.
    test_df.to_csv(result_save_path, index=None, sep=',')


# 处理结果集
def submit_proc(sentence):
    sentence = sentence.lstrip(' ，！。')
    sentence = sentence.replace(' ', '')
    # sentence = sentence.replace('<UNK>', '')
    if sentence == '':
        sentence = '随时联系'
    return sentence


if __name__ == '__main__':
    # 获得参数
    params = get_params()
    # 获得参数
    test(params)