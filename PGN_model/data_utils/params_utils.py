#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2020/3/7 16:05
# @Author : TheTAO
# @Site : 
# @File : params_utils.py
# @Software: PyCharm
import argparse
from PGN_model.data_utils.config import *
from PGN_model.data_utils.file_utils import get_result_filename


# add_argument 变量名、默认值、提示以及类型
def get_params():
    steps_per_epoch = sample_total // batch_size
    # 获得参数
    parser = argparse.ArgumentParser()

    # 模型训练有关
    parser.add_argument("--mode", default='test', help="运行模式", type=str)
    parser.add_argument("--epochs", default=epochs, help="训练周期", type=int)
    parser.add_argument("--min_dec_steps", default=4, help="最小解码步长", type=int)
    parser.add_argument("--max_train_steps", default=500000 / (batch_size / 8), help="最大训练步数", type=int)
    # 开关参数
    parser.add_argument("--pointer_gen", default=True, help="是否使用pointer", type=bool)
    parser.add_argument("--use_coverage", default=True, help="是否使用coverage loss", type=bool)
    parser.add_argument("--greedy_decode", default=False, help="是否使用greedy策略", type=bool)
    parser.add_argument("--bi_gru", default=True, help="是否使用双向GRU", type=bool)
    # 评估与测试参数
    parser.add_argument("--max_num_to_eval", default=50, help="最大评估数量", type=int)
    parser.add_argument("--num_to_test", default=20000, help="测试数量", type=int)
    # 显卡显存分配参数以1000为单位
    parser.add_argument("--gpu_memory", default=6, help="gpu_memory GB", type=int)

    # 优化器与初始化参数有关
    parser.add_argument("--learning_rate", default=0.15, help="学习率参数", type=float)
    parser.add_argument("--adagrad_init_acc", default=0.1, help="Adagrad优化器的初始累加值", type=float)
    parser.add_argument('--rand_unif_init_mag', default=0.02, help='LSTM细胞随机均匀初始化系数', type=float)
    parser.add_argument('--eps', default=1e-12, help='eps', type=float)
    parser.add_argument('--trunc_norm_init_std', default=1e-4,
                        help='std of trunc norm init, used for initializing everything else',
                        type=float)

    # coverage loss权重参数
    parser.add_argument('--cov_loss_wt', default=1.0,
                        help='coverage loss的权重参数 If zero, then no incentive to minimize coverage loss.', type=float)
    # 梯度裁剪参数，防止梯度爆炸
    parser.add_argument('--max_grad_norm', default=2.0, help='for gradient clipping', type=float)

    # 数据预处理相关
    parser.add_argument("--max_enc_len", default=400, help="编码器最大输入长度", type=int)
    parser.add_argument("--max_dec_len", default=50, help="解码器最大输入长度", type=int)
    parser.add_argument("--batch_size", default=batch_size, help="batch size", type=int)
    parser.add_argument("--vocab_path", default=vocab_path, help="vocab path", type=str)
    parser.add_argument("--vocab_size", default=31817, help="词典长度", type=int)
    parser.add_argument("--max_vocab_size", default=50000, help="最大词典长度", type=int)
    parser.add_argument("--beam_size", default=batch_size, help="beam搜索的宽度必须与batch相等", type=int)
    parser.add_argument("--embed_size", default=embedding_dim, help="词向量训练好的维度", type=int)
    # 如果采用双向GRU则需要enc_units/2
    parser.add_argument("--enc_units", default=256, help="Encoder GRU cell units number", type=int)
    parser.add_argument("--dec_units", default=256, help="Decoder GRU cell units number", type=int)
    parser.add_argument("--attn_units", default=256, help="[context vector, decoder state, decoder input] feedforward",
                        type=int)

    # 文件路径有关
    parser.add_argument("--train_seg_x_dir", default=train_x_seg_path, help="train_seg_x_dir", type=str)
    parser.add_argument("--train_seg_y_dir", default=train_y_seg_path, help="train_seg_y_dir", type=str)
    # 验证集存储路径
    parser.add_argument("--val_seg_x_dir", default=val_x_seg_path, help="val_x_seg_path", type=str)
    parser.add_argument("--val_seg_y_dir", default=val_y_seg_path, help="val_y_seg_path", type=str)
    # 测试集存储路径
    parser.add_argument("--test_seg_x_dir", default=test_x_seg_path, help="train_seg_x_dir", type=str)
    # 模型存储路径
    parser.add_argument("--checkpoint_dir", default=checkpoint_dir, help="checkpoint_dir", type=str)
    parser.add_argument("--checkpoints_save_steps", default=5, help="Save checkpoints every N steps", type=int)
    # 测试结果存储路径
    parser.add_argument("--test_save_dir", default=save_result_dir, help="test_save_dir", type=str)
    # 测试结果文件名构建
    parser.add_argument("--result_save_path", default=get_result_filename(batch_size, epochs, 200, 300, 256),
                        help='result_save_path', type=str)
    parser.add_argument("--steps_per_epoch", default=steps_per_epoch, help="max_train_steps", type=int)
    args = parser.parse_args()
    op_param = vars(args)
    return op_param


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
