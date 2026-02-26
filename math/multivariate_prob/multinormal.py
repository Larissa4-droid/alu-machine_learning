#!/usr/bin/env python3
"""Calculates a correlation matrix from a covariance matrix"""
import numpy as np


def correlation(C):
    """
    Calculates a correlation matrix

    Args:
        C: numpy.ndarray of shape (d, d) containing a covariance matrix

    Returns:
        numpy.ndarray of shape (d, d) containing the correlation matrix
    """
    # Validate input type
    if not isinstance(C, np.ndarray):
        raise TypeError("C must be a numpy.ndarray")

    # Validate square matrix shape
    if len(C.shape) != 2 or C.shape[0] != C.shape[1]:
        raise ValueError("C must be a 2D square matrix")

    # Extract the diagonal (variances)
    variances = np.diag(C)

    # Calculate standard deviations (sqrt of variances)
    std_devs = np.sqrt(variances)

    # Create an outer product of std_devs to get a matrix of (std_i * std_j)
    # This facilitates element-wise division: Corr = Cov / (std_i * std_j)
    outer_std = np.outer(std_devs, std_devs)

    # Calculate correlation matrix
    corr = C / outer_std

    return corr
