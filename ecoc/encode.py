#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Date    : Jun-20-20 03:43
# @Author  : Kan HUANG (kan.huang@connect.ust.hk)

import os
import csv
import torch
from ecoc.math_utils import min_max_hamming_distance


def read_codebook(codebook_file):
    """read_codebook
    """
    if not os.path.isfile(codebook_file):
        raise Exception('Unknown codebook_file: %s' % codebook_file)

    codebook = []
    with open(codebook_file) as csvfile:
        reader = csv.reader(csvfile)
        for i, row in enumerate(reader):
            if i == 0:
                continue
            codebook.append(row[-1])

    return codebook


def code_set5() -> list:
    """ECOC code set for N=5
    """
    # 硬编码
    code_set = {0: 0b00111,
                1: 0b01011,
                2: 0b01101,
                3: 0b01110,
                4: 0b10011,
                5: 0b10101,
                6: 0b10110,
                7: 0b11001,
                8: 0b11010,
                9: 0b11100}
    return code_set


def code_set17() -> list:
    """ECOC code set for N=17
    """
    # 硬编码
    code_set = {0: 0b00000000011111111,
                1: 0b00000011100011111,
                2: 0b00000011111100011,
                3: 0b00000101101101101,
                4: 0b00000101110110110,
                5: 0b00000110101111010,
                6: 0b00000110111010101,
                7: 0b00000111010111001,
                8: 0b00000111011001110,
                9: 0b00001001111011001}
    return code_set


def code_set23() -> list:
    """ECOC code set for N=23
    """
    # 硬编码
    code_set = {0: 0b00000000000011111111111,
                1: 0b00000001111100000111111,
                2: 0b00000001111111111000001,
                3: 0b00000110011100111001110,
                4: 0b00000110101111001110010,
                5: 0b00000111110001110110100,
                6: 0b00001011001110110111000,
                7: 0b00001011110011001001110,
                8: 0b00010101010111010011010,
                9: 0b00010101100110101101100}
    return code_set


def get_codebook_tensor(codebook):
    """get_codebook_tensor
    """
    num_classes, len_code = len(codebook), len(codebook[0])
    codebook_tensor = torch.zeros(num_classes, len_code)
    for i in range(num_classes):
        for j in range(len_code):
            # codebook_tensor[i][0] stands for classifier 0's output on class i
            codebook_tensor[i][j] += int(codebook[i][j])

    return codebook_tensor


def main():
    codebook_name = "hunqun_deng_c10_n5"
    codebook_file = f"ecoc/codebooks/{codebook_name}.csv"
    codebook = read_codebook(codebook_file)
    print(codebook)

    codebook_tensor = get_codebook_tensor(codebook)
    print(codebook_tensor)
    print(codebook_tensor[:, 0])


def test():
    code_set = code_set23()
    min_d, max_d = min_max_hamming_distance(code_set=code_set)
    print(min_d, max_d)


def test2():
    # 为了加速，直接使用 tensor 数据结构 TODO
    pass


def _test():
    test()


if __name__ == "__main__":
    main()
