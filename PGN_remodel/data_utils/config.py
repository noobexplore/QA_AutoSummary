#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2020/3/7 16:02
# @Author : TheTAO
# @Site : 
# @File : config.py
# @Software: PyCharm
import os
import pathlib

# 预处理数据 构建数据集
is_build_dataset = True

# 获取工程的根目录, 这样路径就可以不区分操作系统
root = pathlib.Path(os.path.abspath(__file__)).parent.parent

# 原始数据保存路径
train_data_path = os.path.join(root, 'datasets', 'AutoMaster_TrainSet.csv')
test_data_path = os.path.join(root, 'datasets', 'AutoMaster_TestSet.csv')

# 停用词的保存路径
stop_word_path = os.path.join(root, 'datasets', 'stopwords', 'stopwords.txt')

# 分词用户自定义字典
user_dict = os.path.join(root, 'datasets', 'cutDict', 'user_dict.txt')

# 预处理后的训练集和测试集的csv文件分词路径
train_seg_path = os.path.join(root, 'datasets', 'train_data', 'train_seg.csv')
test_seg_path = os.path.join(root, 'datasets', 'test_data', 'test_seg.csv')
# 合并后的分词文件路径
merger_seg_path = os.path.join(root, 'datasets', 'merged_train_test_seg_data.csv')

# 1.数据标签分离
train_x_seg_path = os.path.join(root, 'datasets', 'train_data', 'train_X_seg_data.csv')
train_y_seg_path = os.path.join(root, 'datasets', 'train_data', 'train_Y_seg_data.csv')
# 验证数据集
val_x_seg_path = os.path.join(root, 'datasets', 'train_data', 'val_X_seg_data.csv')
val_y_seg_path = os.path.join(root, 'datasets', 'train_data', 'val_Y_seg_data.csv')
# 测试数据集
test_x_seg_path = os.path.join(root, 'datasets', 'test_data', 'test_X_seg_data.csv')

# 2.填充好的训练集和测试集文件
train_x_pad_path = os.path.join(root, 'datasets', 'train_data', 'train_X_pad_data.csv')
train_y_pad_path = os.path.join(root, 'datasets', 'train_data', 'train_Y_pad_data.csv')
test_x_pad_path = os.path.join(root, 'datasets', 'test_data', 'test_X_pad_data.csv')

# 3.numpy转换好的数据
train_x_path = os.path.join(root, 'datasets', 'train_data', 'train_X')
train_y_path = os.path.join(root, 'datasets', 'train_data', 'train_Y')
test_x_path = os.path.join(root, 'datasets', 'test_data', 'test_X')

# 4.词向量保存路径和迭代次数
save_wv_model_path = os.path.join(root, 'datasets', 'wv_model', 'w2v.model')
# 词向量矩阵路径
embedding_matrix_path = os.path.join(root, 'datasets', 'wv_model', 'embedding_matrix.npy')
# 正反向词典路径
vocab_path = os.path.join(root, 'datasets', 'vocab', 'vocab.txt')
reverse_vocab_path = os.path.join(root, 'datasets', 'vocab', 'reverse_vocab.txt')
# 词向量训练轮数
wv_train_epochs = 10

# 5.保存点路径
checkpoint_dir = os.path.join(root, 'datasets', 'checkpoints',
                              'trained_checkpoints_bc16_ed300_lr015_ut256_ep30_vbsize32276')
checkpoint_prefix = os.path.join(checkpoint_dir, 'ckpt')

# 6.保存结果路径
save_result_dir = os.path.join(root, 'result')

# vocabsize = 32272

