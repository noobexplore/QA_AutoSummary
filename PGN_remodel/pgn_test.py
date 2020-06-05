#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/5/30 9:57
# @Author  : TheTao
# @Site    : 
# @File    : pgn_test.py
# @Software: PyCharm
import pandas as pd
import tensorflow as tf
from tqdm import tqdm
from PGN_remodel.PGN_model import PGN
from PGN_remodel.bacher import batcher
from PGN_remodel.data_utils.config import *
from PGN_remodel.data_utils.wv_loader import Vocab
from PGN_remodel.data_utils.params_utils import get_params
from PGN_remodel.pgn_test_helper import greedy_decode
from PGN_remodel.pgn_beam_test_helper import beam_decode


def test(params):
    assert params["mode"].lower() in ["test", "eval"], "change training mode to 'test' or 'eval'"
    # assert params["beam_size"] == params["batch_size"], "Beam size must be equal to batch_size, change the params"
    print("Building the model ...")
    model = PGN(params)
    print("Creating the vocab ...")
    vocab = Vocab(params["vocab_path"], params["vocab_size"])
    print("Creating the checkpoint manager")
    checkpoint = tf.train.Checkpoint(PGN=model)
    checkpoint_manager = tf.train.CheckpointManager(checkpoint, checkpoint_dir, max_to_keep=5)
    checkpoint.restore(checkpoint_manager.latest_checkpoint)
    if checkpoint_manager.latest_checkpoint:
        print("Restored from {}".format(checkpoint_manager.latest_checkpoint))
    else:
        print("Initializing from scratch.")
    print("Model restored")
    if params['greedy_decode']:
        predict_greedy_result(model, params, vocab, params['result_save_path'])
    else:
        b = batcher(vocab, params)
        for batch in b:
            yield beam_decode(model, batch, vocab, params)


def predict_greedy_result(model, params, vocab, result_save_path):
    dataset = batcher(vocab, params)
    # 预测结果
    results = greedy_decode(model, dataset, vocab, params)
    results = list(map(lambda x: x.replace(" ", ""), results))
    # 保存结果
    save_predict_result(results, result_save_path)
    return results


# beamsearch
def beamsearch_test(params):
    gen = test(params)
    results = []
    with tqdm(total=params["num_to_test"], position=0, leave=True) as pbar:
        for i in range(params["num_to_test"]):
            trial = next(gen)
            results.append(trial.abstract)
            pbar.update(1)
    results = list(map(lambda x: x.replace(" ", ""), results))
    save_predict_result(results, params['result_save_path'])


def save_predict_result(results, result_save_path):
    # 读取结果
    test_df = pd.read_csv(test_data_path)
    print('result len: {}'.format(len(results)))
    # 填充结果
    test_df['Prediction'] = results
    # 提取ID和预测结果两列
    test_df = test_df[['QID', 'Prediction']]
    # 批量去掉空格和特殊字符
    test_df['Prediction'] = test_df['Prediction'].apply(submit_proc)
    # 保存结果
    test_df.to_csv(result_save_path, index=None, sep=',')


# 处理结果集
def submit_proc(sentence):
    sentence = sentence.replace(' ', '')
    sentence = sentence.replace('<UNK>', '')
    if sentence == '':
        sentence = '随时联系'
    return sentence


# 文件去除空格函数
def clean_results(file_path):
    result_df = pd.read_csv(file_path)
    result_df['Prediction'] = result_df['Prediction'].apply(lambda x: x.replace(" ", ""))
