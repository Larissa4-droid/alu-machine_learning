#!/usr/bin/env python3
"""
Contains the function correlation that calculates a correlation matrix
"""
import numpy as np


def correlation(C):
    """
    Calculates a correlation matrix from a covariance matrix
    Args:
        C: numpy.ndarray of shape (d, d) containing a covariance matrix
    Returns:
        numpy.ndarray of shape (d, d) containing the correlation matrix
    """
    if not isinstance(C, np.ndarray):
        raise TypeError("C must be a numpy.ndarray")

    if len(C.shape) != 2 or C.shape[0] != C.shape[1]:
        raise ValueError("C must be a 2D square matrix")

    # Extract the diagonal elements (variances)
    diag = np.diag(C)

    # Calculate standard deviations (square root of variances)
    std_dev = np.sqrt(diag)

    # Create the outer product of standard deviations (std_i * std_j)
    # This creates the denominator for every element in the matrix
    normalization_matrix = np.outer(std_dev, std_dev)

    # Calculate the correlation matrix: Cov / (std_i * std_j)
    corr = C / normalization_matrix

    return corr
