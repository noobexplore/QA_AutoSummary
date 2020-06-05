#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2020/3/7 16:05
# @Author : TheTAO
# @Site : 
# @File : params_utils.py
# @Software: PyCharm
import argparse
from seq2seq_model_v2.data_utils.config import vocab_path, train_x_seg_path, train_y_seg_path, test_x_seg_path, \
    sample_total, batch_size, save_result_dir, epochs, checkpoint_dir, vocab_size, lstm_units
from seq2seq_model_v2.data_utils.file_utils import get_result_filename


# add_argument 变量名、默认值、提示以及类型
def get_params():
    parser = argparse.ArgumentParser()
    # 默认的模式参数，默认为测试模式
    parser.add_argument('--mode', default='test', help='run mode', type=str)
    # 默认的最大编码输入长度
    parser.add_argument('--max_enc_len', default=200, help='Encoder input max sequence length', type=int)
    # 默认的最大解码输入长度
    parser.add_argument('--max_dec_len', default=41, help='Decoder input max sequence length', type=int)
    # 默认的batch_size长度，这里batch_size为了后面与beamsearch调试方便
    parser.add_argument('--batch_size', default=batch_size, help='batch size', type=int)
    # 默认的词典路径
    parser.add_argument('--vocab_path', default=vocab_path, help='vocab path', type=str)
    # 默认的词表大小
    parser.add_argument("--vocab_size", default=vocab_size, help="max vocab size , None-> Max ", type=int)
    # 默认的学习率的设置
    parser.add_argument('--learning_rate', default=0.0001, help='Learning rate', type=float)
    # 默认的adagrad初始化计算精度
    parser.add_argument("--adagrad_init_acc", default=0.1,
                        help="Adagrad optimizer initial accumulator value. "
                             "Please refer to the Adagrad optimizer API documentation "
                             "on tensorflow site for more details.",
                        type=float)
    # 默认的lSTM细胞初始化量级为随机均匀初始化
    parser.add_argument('--rand_unif_init_mag', default=0.02,
                        help='magnitude for lstm cells random uniform inititalization', type=float)
    # 默认的截断高斯正态分布的初始化系数
    parser.add_argument('--trunc_norm_init_std', default=1e-4,
                        help='std of trunc norm init, used for initializing everything else',
                        type=float)
    # 默认的coverage_loss系数
    parser.add_argument('--cov_loss_wt', default=1.0, help='Weight of coverage loss (lambda in the paper).'
                                                           ' If zero, then no incentive to minimize coverage loss.',
                        type=float)
    # 默认的梯度裁剪系数
    parser.add_argument('--max_grad_norm', default=2.0, help='for gradient clipping', type=float)
    # 设置beam search的beam的范围
    parser.add_argument("--beam_size", default=batch_size,
                        help="beam size for beam search decoding (must be equal to batch size in decode mode)",
                        type=int)
    # 设置embedd_size、enc_utils、dec_utils和attn_utils
    parser.add_argument("--embed_size", default=300, help="Words embeddings dimension", type=int)
    parser.add_argument("--enc_units", default=lstm_units, help="Encoder GRU cell units number", type=int)
    parser.add_argument("--dec_units", default=lstm_units, help="Decoder GRU cell units number", type=int)
    parser.add_argument("--attn_units", default=lstm_units, help="[context vector, decoder state, decoder input] "
                                                                 "feedforward \
                                result dimension - this result is used to compute the attention weights",
                        type=int)
    # 设置处理好的训练集数据以及测试集数据的默认路径
    parser.add_argument("--train_seg_x_dir", default=train_x_seg_path, help="train_seg_x_dir", type=str)
    parser.add_argument("--train_seg_y_dir", default=train_y_seg_path, help="train_seg_y_dir", type=str)
    parser.add_argument("--test_seg_x_dir", default=test_x_seg_path, help="train_seg_x_dir", type=str)
    # 设置模型训练的默认路径
    parser.add_argument("--checkpoint_dir", default=checkpoint_dir, help="checkpoint_dir", type=str)
    # 默认的几个步骤开始保存
    parser.add_argument("--checkpoints_save_steps", default=5, help="Save checkpoints every N steps", type=int)
    # 默认的最小解码步长
    parser.add_argument("--min_dec_steps", default=4, help="min_dec_steps", type=int)
    # 默认最大训练步长，样本总大小除以批量大小
    parser.add_argument("--max_train_steps", default=sample_total // batch_size, help="max_train_steps", type=int)
    # 默认设置是否每个批量保存训练数据
    parser.add_argument("--save_batch_train_data", default=False, help="save batch train data to pickle", type=bool)
    parser.add_argument("--load_batch_train_data", default=False, help="load batch train data from pickle",
                        type=bool)
    # 默认测试结果保存路径
    parser.add_argument("--test_save_dir", default=save_result_dir, help="test_save_dir", type=str)
    # 设置是否使用PGN网络，默认为否
    parser.add_argument("--pointer_gen", default=False, help="training, eval or test options", type=bool)
    # 设置是否使用coverage_loss
    parser.add_argument("--use_coverage", default=False, help="test_save_dir", type=bool)
    # 设置是否使用贪婪解码方式
    parser.add_argument("--greedy_decode", default=True, help="greedy_decode", type=bool)
    # 获取默认的结果文件名
    parser.add_argument("--result_save_path", default=get_result_filename(batch_size, epochs, 300, 300, 512),
                        help='result_save_path', type=str)
    # 默认训练周期
    parser.add_argument("--epochs", default=epochs, help="train of epochs", type=int)

    args = parser.parse_args()
    params = vars(args)
    return params


# 直接获取到默认的参数
def get_default_params():
    params = {"mode": 'train',
              "max_enc_len": 300,
              "max_dec_len": 41,
              "batch_size": batch_size,
              "epochs": 25,
              "vocab_path": vocab_path,
              "learning_rate": 0.0001,
              "adagrad_init_acc": 0.1,
              "rand_unif_init_mag": 0.02,
              "trunc_norm_init_std": 1e-4,
              "cov_loss_wt": 1.0,
              "max_grad_norm": 2.0,
              "vocab_size": vocab_size,
              "beam_size": batch_size,
              "embed_size": 300,
              "enc_units": 256,
              "dec_units": 256,
              "attn_units": 256,
              "train_seg_x_dir": train_x_seg_path,
              "train_seg_y_dir": train_y_seg_path,
              "test_seg_x_dir": test_x_seg_path,
              "checkpoints_save_steps": 5,
              "min_dec_steps": 4,
              "max_train_steps": sample_total // batch_size,
              "train_pickle_dir": './datasets/pickle',
              "save_batch_train_data": False,
              "load_batch_train_data": False,
              "test_save_dir": save_result_dir,
              "pointer_gen": False,
              "use_coverage": False
              }
    return params
