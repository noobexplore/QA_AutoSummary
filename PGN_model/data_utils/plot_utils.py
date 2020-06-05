#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2020/3/7 16:08
# @Author : TheTAO
# @Site : 
# @File : plot_utils.py
# @Software: PyCharm
from matplotlib import font_manager
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt

# 解决中文乱码
font = font_manager.FontProperties(fname="./seq2seq_model_v2/data_utils/font/TrueType/simhei.ttf")


# 根据权重矩阵画出对应的像素矩阵
def plot_attention(attention, sentence, predicted_sentence):
    fig = plt.figure(figsize=(20, 20))
    ax = fig.add_subplot(1, 1, 1)
    ax.matshow(attention, cmap='viridis')
    fontdict = {'fontsize': 12, 'fontproperties': font}
    ax.set_xticklabels([''] + sentence, fontdict=fontdict, rotation=90)
    ax.set_yticklabels([''] + predicted_sentence, fontdict=fontdict)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
    plt.show()