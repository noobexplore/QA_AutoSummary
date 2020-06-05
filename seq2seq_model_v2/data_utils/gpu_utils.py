#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2020/3/7 16:09
# @Author : TheTAO
# @Site : 
# @File : gpu_utils.py
# @Software: PyCharm
import tensorflow as tf


def config_gpu():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
                logical_gpus = tf.config.experimental.list_logical_devices('GPU')
                print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            print(e)


if __name__ == '__main__':
    config_gpu()