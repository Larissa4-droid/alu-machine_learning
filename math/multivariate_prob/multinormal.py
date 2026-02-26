#!/usr/bin/env python3
"""
Defines the MultiNormal class
"""
import numpy as np


class MultiNormal:
    """
    Represents a Multivariate Normal distribution
    """

    def __init__(self, data):
        """
        Calculates the mean and covariance of a data set
        Args:
            data: numpy.ndarray of shape (d, n) containing the data set
        """
        if not isinstance(data, np.ndarray) or len(data.shape) != 2:
            raise TypeError("data must be a 2D numpy.ndarray")

        d, n = data.shape

        if n < 2:
            raise ValueError("data must contain multiple data points")

        # Calculate mean: average along axis 1 (columns)
        # keepdims=True ensures the shape is (d, 1)
        self.mean = np.mean(data, axis=1, keepdims=True)

        # Center the data by subtracting the mean
        # (d, n) - (d, 1) broadcasts correctly across all n points
        data_centered = data - self.mean

        # Calculate covariance: (X_centered @ X_centered.T) / (n - 1)
        # (d, n) matrix multiplied by (n, d) results in (d, d)
        self.cov = np.dot(data_centered, data_centered.T) / (n - 1)
