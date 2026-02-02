#!/usr/bin/env python3
"""Module for element-wise operations on numpy arrays"""


def np_elementwise(mat1, mat2):
    """Performs element-wise operations on matrices"""
    return (mat1 + mat2, mat1 - mat2, mat1 * mat2, mat1 / mat2)