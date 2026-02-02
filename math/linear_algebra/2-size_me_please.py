#!/usr/bin/env python3
"""Module for calculating matrix shape"""


def matrix_shape(matrix):
    """Calculates the shape of a matrix"""
    shape = []
    while isinstance(matrix, list):
        shape.append(len(matrix))
        matrix = matrix[0]
    return shape


