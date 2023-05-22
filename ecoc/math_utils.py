#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Date    : Jun-20-20 03:43
# @Author  : Kan HUANG (kan.huang@connect.ust.hk)

import numpy as np


def count_ones(n: int):
    """count_ones
    """
    count = 0
    while count < n:
        n &= n-1  # 清除最低位的1
        count += 1
    return count


def hamming_distance(x: int, y: int) -> int:
    """hamming_distance
    """
    n = x ^ y
    count = count_ones(n)
    return count


def min_max_hamming_distance(codebook: list) -> tuple:
    """min_max_hamming_distance
    The min and max hamming distance between all code pairs in a codebook.
    """
    num_codes = len(codebook)
    # 128 bits' difference is large enough
    distances = np.zeros([num_codes, num_codes]) + 128
    for i in range(num_codes):
        for j in range(num_codes):
            distances[i][j] = hamming_distance(codebook[i], codebook[j])

    max_hamming_distance = int(np.max(distances))
    for i in range(num_codes):
        distances[i][i] = 128
    min_hamming_distance = int(np.min(distances))

    return min_hamming_distance, max_hamming_distance
