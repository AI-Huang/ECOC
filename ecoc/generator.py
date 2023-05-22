#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Date    : Jun-18-20 20:18
# @Author  : Kan HUANG (kan.huang@connect.ust.hk)

import os
import yaml
import time
from ecoc.math_utils import count_ones, hamming_distance, min_max_hamming_distance


def get_valid_codes(num_codes: int, num_ones: int) -> list:
    """get_valid_codes
    Args:
        num_codes
        num_ones:

    Return:

    Paper: Applying Error-Correcting Output Coding to Enhance Convolutional Neural Network for Target Detection and Pattern Recognition, https://ieeexplore.ieee.org/document/5597751
    """
    codes = []
    for c in range(num_codes):  # 0b00...0 ~ 0b11...1
        if count_ones(c) == num_ones:  # Must contain M 1's
            codes.append(c)
    return codes


def get_column_codebook(row_codebook: list, len_code: int):
    """get_column_codebook
    Each column code stands for a classifier function $f_i$. $f_i$ should be uncorrelated with the functions on other bits $f_j$, $j\neq i$.

    Args:
        row_codebook: list,
        len_code: int, length of the row code.

    Return:
        codebook: last code for last classifier; high bit stands for class with larger index

    Paper: Solving Multiclass Learning Problems via Error-Correcting Output Codes, https://www.jair.org/index.php/jair/article/view/10127
    """
    row_codebook = row_codebook.copy()
    codebook = []
    for _ in range(len_code):
        column_code = 0b0
        for j in range(len(row_codebook)):
            # Little-endian mode
            column_code += (row_codebook[j] & 1) << j
            # Next bit position
            row_codebook[j] = row_codebook[j] >> 1
        codebook.insert(0, column_code)

    return codebook


def generate_codebook(num_classes: int,
                      len_code: int,
                      num_ones: int,
                      min_hamming_distance: int) -> list:
    """generate_codebook
    Args:

    Return:
        codebook:

    Paper: Applying Error-Correcting Output Coding to Enhance Convolutional Neural Network for Target Detection and Pattern Recognition, https://ieeexplore.ieee.org/document/5597751
    """
    N, M, D = len_code, num_ones, min_hamming_distance
    codes = get_valid_codes(num_codes=2 << N, num_ones=M)

    count, codebook = 0, []
    while count < num_classes:
        # Choose one by one 贪婪法
        is_found = False
        # 顺序查找，结果会与初始选择关系较大
        for c in codes:
            # Validation
            is_valid = True
            for c2 in codebook:
                if hamming_distance(c, c2) >= D:
                    pass
                else:
                    codes.remove(c)
                    is_valid = False
                    break
            if is_valid:
                # Appending
                codebook.append(c)
                count += 1
                codes.remove(c)
                is_found = True
                break
        if not is_found:
            print("Failed finding all codes.")
            break

    return codebook


def main():
    os.chdir(__file__.rsplit(os.sep, 1)[0] + os.sep)
    with open("./config/config_c10.yaml", encoding='UTF-8') as yaml_file:
        config = yaml.safe_load(yaml_file)
    num_classes = config["num_classes"]
    config.pop("num_classes")

    all_codebooks = list(config)
    codebook_name = all_codebooks[0]  # Modify here to choose the codebook
    print(f"codebook_name: {codebook_name}")
    params = config[codebook_name]

    start = time.process_time()

    len_code = params["len_code"]
    num_ones = params["num_ones"]
    min_hamming_distance = params["min_hamming_distance"]
    expected_max_hamming_distance = params["max_hamming_distance"]
    codebook = generate_codebook(
        num_classes, len_code, num_ones, min_hamming_distance
    )

    elapsed = (time.process_time() - start)
    print(f"Time used: {elapsed}s.")

    for i, c in enumerate(codebook):
        print(f"Code {i}: {c:0{len_code}b}")
    # Min and max hamming distance
    min_hamming_distance, max_hamming_distance = min_max_hamming_distance(
        codebook)
    print(
        f"max_hamming_distance: {max_hamming_distance}", f"expected_max_hamming_distance: {expected_max_hamming_distance}")

    with open(f"./codebooks/{codebook_name}.csv", "w") as f:
        f.write("class,codeword"+'\n')
        for i, c in enumerate(codebook):
            f.write(f"{i},{c:0{len_code}b}"+'\n')

    column_codebook = get_column_codebook(codebook, len_code)
    for i, c in enumerate(column_codebook):
        print(f"{i},{c:0{num_classes}b}")
    min_hamming_distance, max_hamming_distance = min_max_hamming_distance(
        column_codebook)
    print(
        f"min_hamming_distance: {min_hamming_distance}", f"max_hamming_distance: {max_hamming_distance}")


if __name__ == "__main__":
    main()
