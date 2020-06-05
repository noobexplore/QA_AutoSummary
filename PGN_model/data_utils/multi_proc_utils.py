#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2020/3/7 16:07
# @Author : TheTAO
# @Site : 
# @File : mutil_proc_utils.py
# @Software: PyCharm
import pandas as pd
import numpy as np
from multiprocessing import cpu_count, Pool

# cpu 数量
cores = cpu_count()
# 分块个数
partitions = cores


def parallelize(df, func):
    """
    多核并行处理模块
    :param df: DataFrame数据
    :param func: 预处理函数
    :return: 处理后的数据
    """
    # 数据切分
    df_split = np.array_split(df, partitions)
    # 线程池
    pool = Pool(cores)
    # 数据的分发合并
    data = pd.concat(pool.map(func, df_split))
    # 关闭线程池
    pool.close()
    # 执行完close后，如果没新的线程池加入的话join自动结束所有进程
    pool.join()
    return data