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
        self.mean = np.mean(data, axis=1, keepdims=True)
        data_centered = data - self.mean
        self.cov = np.dot(data_centered, data_centered.T) / (n - 1)

    def pdf(self, x):
        """
        Calculates the PDF at a data point
        Args:
            x: numpy.ndarray of shape (d, 1) containing the data point
        Returns:
            The value of the PDF
        """
        if not isinstance(x, np.ndarray):
            raise TypeError("x must be a numpy.ndarray")

        d = self.mean.shape[0]
        if len(x.shape) != 2 or x.shape[0] != d or x.shape[1] != 1:
            raise ValueError("x must have the shape ({}, 1)".format(d))
        # Det and Inv of the covariance matrix
        det = np.linalg.det(self.cov)
        inv = np.linalg.inv(self.cov)
        # Normalization constant: 1 / sqrt((2*pi)^d * det)
        norm_const = 1 / np.sqrt(((2 * np.pi) ** d) * det)
        # Mahalanobis distance part: (x - mu).T @ inv @ (x - mu)
        diff = x - self.mean
        exponent = -0.5 * np.dot(np.dot(diff.T, inv), diff)
        # Combine results
        pdf_value = norm_const * np.exp(exponent)
        # Return as a scalar float rather than a 1x1 array
        return pdf_value[0][0]
