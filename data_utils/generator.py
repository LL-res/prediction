import random

import numpy as np

#这个base的周期是12
default_base = [30, 21, 29, 31, 40, 48, 53, 47, 37, 39, 31, 29, 17, 9, 20, 24, 27, 35, 41, 38,
                27, 31, 27, 26, 21, 13, 21, 18, 33, 35, 40, 36, 22, 24, 21, 20, 17, 14, 17, 19,
                26, 29, 40, 31, 20, 24, 18, 26, 17, 9, 17, 21, 28, 32, 46, 33, 23, 28, 22, 27,
                18, 8, 17, 21, 31, 34, 44, 38, 31, 30, 26, 32]

def generate(base,  random_range, amplify_factor=1,total_size = -1):
    """
    Parameters:
      base - 传入空list则使用默认的序列
      random_range - list中的元素被加的随机范围，加入的随机值范围为 -random_range ～ random_range
      amplify_factor - list中每个元素被放大的倍数
      total_size - 如果为-1,则传入的序列不会被加长，否则会被加长或缩短到total_size

    Returns:
        可用于预测或训练的序列

    """
    total_set_raw = []
    if len(base) == 0:
        base = default_base
    if total_size == -1:
        for i in range(len(base)):
            element = amplify_factor * (base[i] + random.randint(-random_range, random_range))
            total_set_raw.append(element)
    else:
        for i in range(total_size):
            element = amplify_factor * (base[i % len(base)] + random.randint(-random_range, random_range))
            total_set_raw.append(element)
    return total_set_raw
