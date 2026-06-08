# -*- coding: utf-8 -*-
"""
    @Author  : Zephyr
    @Date    : 2026/6/2 15:48
    @Project  : deeplearningproject
    @File     : 1.py
    @IDE      : PyCharm
    @Description: 
"""

# 读取common data数据，返回训练集和测试集
import os

import numpy as np
import pandas as pd

def read_data():
    train_data = pd.read_csv(os.path.join(os.path.dirname(__file__), 'data', 'train.csv'))
    test_data = pd.read_csv(os.path.join(os.path.dirname(__file__), 'data', 'test.csv'))
    return train_data, test_data
