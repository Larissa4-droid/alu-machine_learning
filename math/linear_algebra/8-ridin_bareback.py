#!/usr/bin/env python3
"""
Module for matrix multiplication
"""


def mat_mul(mat1, mat2):
    """Performs matrix multiplication on two 2D matrices"""
    if len(mat1[0]) != len(mat2):
        return None

    result = []
    for i in range(len(mat1)):
        new_row = []
        for j in range(len(mat2[0])):
            elem = 0
            for k in range(len(mat2)):
                elem += mat1[i][k] * mat2[k][j]
            new_row.append(elem)
        result.append(new_row)
    return result
