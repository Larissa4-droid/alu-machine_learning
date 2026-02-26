#!/usr/bin/env python3
"""Exponential distribution module.

This module contains a class that represents an Exponential distribution.
"""


class Exponential:
    """Represents an Exponential distribution."""

    def __init__(self, data=None, lambtha=1.):
        """Initialize an Exponential distribution.

        Args:
            data (list): list of data to estimate the distribution parameter
            lambtha (float): expected number of occurrences

        Raises:
            ValueError: if lambtha is not a positive value when data is None
            TypeError: if data is not a list when provided
            ValueError: if data contains fewer than two values
        """
        if data is None:
            if lambtha <= 0:
                raise ValueError("lambtha must be a positive value")
            self.lambtha = float(lambtha)
        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")

            mean = sum(data) / len(data)
            self.lambtha = float(1 / mean)

    def pdf(self, x):
        """Calculates the PDF value for x."""
        if x < 0:
            return 0
        e = 2.7182818285
        return self.lambtha * (e ** (-self.lambtha * x))

    def cdf(self, x):
        """Calculates the CDF value for x."""
        if x < 0:
            return 0
        e = 2.7182818285
        return 1 - (e ** (-self.lambtha * x))
