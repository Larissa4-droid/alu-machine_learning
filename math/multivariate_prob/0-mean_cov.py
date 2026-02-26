#!/usr/bin/env python3
"""Calculates the mean and covariance of a data set"""
import numpy as np


def mean_cov(X):
    """
    Calculates the mean and covariance of a data set

    Args:
        X: numpy.ndarray of shape (n, d) containing the data set

    Returns:
        mean: numpy.ndarray of shape (1, d) containing the mean
        cov: numpy.ndarray of shape (d, d) containing the covariance matrix
    """
    # Validate input type and dimensions
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        raise TypeError("X must be a 2D numpy.ndarray")

    n, d = X.shape

    # Validate number of data points
    if n < 2:
        raise ValueError("X must contain multiple data points")

    # Calculate mean along the columns (axis 0)
    # keepdims=True ensures the shape is (1, d)
    mean = np.mean(X, axis=0, keepdims=True)

    # Center the data by subtracting the mean
    X_centered = X - mean

    # Calculate covariance: (X_centered.T @ X_centered) / (n - 1)
    # Using dot product for efficiency
    cov = np.dot(X_centered.T, X_centered) / (n - 1)

    return mean, cov
