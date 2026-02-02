#!/usr/bin/env python3
"""
Module to concatenate 2D matrices
"""


def cat_matrices2D(mat1, mat2, axis=0):
    """Concatenates two 2D matrices along a specific axis"""
    if axis == 0:
        # Concatenate along rows (vertically)
        # Check if both matrices have the same number of columns
        if len(mat1[0]) != len(mat2[0]):
            return None
        # Create a new matrix by copying rows from both matrices
        return [row[:] for row in mat1] + [row[:] for row in mat2]
    elif axis == 1:
        # Concatenate along columns (horizontally)
        # Check if both matrices have the same number of rows
        if len(mat1) != len(mat2):
            return None
        # Create a new matrix by concatenating corresponding rows
        return [mat1[i][:] + mat2[i][:] for i in range(len(mat1))]
    else:
        return None
    