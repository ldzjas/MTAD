# -*- coding: utf-8 -*-
"""
读取 ./WaDi/processing/ 下的 train.pkl、test.pkl、test_label.pkl 并打印形状。
运行：
    python ./WaDi/readpkl.py
"""

import os
import pickle
import numpy as np

PROCESS_DIR = os.path.join('.', 'SMD')
TRAIN_PKL = os.path.join(PROCESS_DIR, 'machine-1-1_train.pkl')
TEST_PKL = os.path.join(PROCESS_DIR, 'machine-1-1_test.pkl')
TEST_LABEL_PKL = os.path.join(PROCESS_DIR, 'machine-1-1_test_label.pkl')


def load_pkl(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f'文件不存在: {path}')
    with open(path, 'rb') as f:
        obj = pickle.load(f)
    return obj


def to_numpy(x):
    return x if isinstance(x, np.ndarray) else np.array(x)


def main():
    print('读取并打印 pkl 形状...')
    train = to_numpy(load_pkl(TRAIN_PKL))
    test = to_numpy(load_pkl(TEST_PKL))
    labels = to_numpy(load_pkl(TEST_LABEL_PKL))

    print(f'train.pkl 形状: {train.shape}')
    print(f'test.pkl  形状: {test.shape}')
    # 标签一般为一维向量，同时打印长度以便检查
    print(f'test_label.pkl 形状: {labels.shape}，长度: {labels.shape[0] if labels.ndim > 0 else 0}')

    # 可选：一致性检查
    if test.shape[0] != labels.shape[0]:
        print(f'警告：测试样本数({test.shape[0]})与标签数({labels.shape[0]})不一致。')


if __name__ == '__main__':
    main()
