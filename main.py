#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2020/3/7 17:32
# @Author : TheTAO
# @Site : 
# @File : main.py
# @Software: PyCharm
import warnings
import json as js
from PGN_remodel.pgn_train import train
from PGN_remodel.pgn_test import test, beamsearch_test
from PGN_remodel.data_utils.params_utils import get_params

warnings.filterwarnings("ignore")

if __name__ == '__main__':
    # 加载参数
    params = get_params()
    # 默认模式
    params['mode'] = 'test'
    # 训练重要参数
    params['epochs'] = 30
    # 最好为整数不然bacher那里会出问题
    params['batch_size'] = 3
    # test的重要选项
    params['greedy_decode'] = False
    params['decode_mode'] = 'beam'
    # beamsearch的时候一定要设置
    params['beam_size'] = 3
    # 格式化输出params
    js_params = js.dumps(params, ensure_ascii=False, indent=4, separators=(',', ': '))
    # 将参数存入临时文件以便查看对比
    with open('./PGN_remodel/paramsjson/params.json', 'w', encoding='utf-8') as f:
        f.write(js_params)
    # 运行模型
    if params['mode'] == 'train':
        train(params)
    elif params['mode'] == 'test':
        if params['decode_mode'] == 'beam':
            beamsearch_test(params)
        else:
            test(params)
    else:
        print('eval')
