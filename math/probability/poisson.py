#!/usr/bin/env python3
"""Poisson distribution module.

This module contains a class that represents a Poisson distribution.
"""


class Poisson:
    """Represents a Poisson distribution."""

    def __init__(self, data=None, lambtha=1.):
        """Initialize a Poisson distribution.

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
            self.lambtha = float(sum(data) / len(data))

    def pmf(self, k):
        """Calculates the PMF for a given number of successes.

        Args:
            k (int): number of successes

        Returns:
            float: PMF value for k
        """
        k = int(k)

        if k < 0:
            return 0

        # factorial of k
        factorial = 1
        for i in range(1, k + 1):
            factorial *= i

        # approximate e^(-lambtha)
        e = 2.7182818285
        numerator = (self.lambtha ** k) * (e ** (-self.lambtha))
        return numerator / factorial

    def cdf(self, k):
        """Calculates the CDF for a given number of successes.

        Args:
            k (int): number of successes

        Returns:
            float: CDF value for k
        """
        k = int(k)
        if k < 0:
            return 0

        cdf_sum = 0
        for i in range(0, k + 1):
            cdf_sum += self.pmf(i)

        return cdf_sum
