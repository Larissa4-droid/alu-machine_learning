#!/usr/bin/env python3
"""Normal distribution module.

This module contains a class that represents a Normal distribution.
"""


class Normal:
    """Represents a Normal distribution."""

    def __init__(self, data=None, mean=0., stddev=1.):
        """Initialize a Normal distribution.

        Args:
            data (list): list of data to estimate the distribution
            mean (float): mean of the distribution
            stddev (float): standard deviation of the distribution

        Raises:
            ValueError: if stddev is not a positive value when dt is None
            TypeError: if data is not a list when provided
            ValueError: if data contains fewer than two values
        """
        if data is None:
            if stddev <= 0:
                raise ValueError("stddev must be a positive value")
            self.mean = float(mean)
            self.stddev = float(stddev)
        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")

            n = len(data)
            mean_value = sum(data) / n

            variance = 0
            for x in data:
                variance += (x - mean_value) ** 2
            variance /= n

            self.mean = float(mean_value)
            self.stddev = float(variance ** 0.5)

    def z_score(self, x):
        """Calculates the z-score of a given x-value."""
        return (x - self.mean) / self.stddev

    def x_value(self, z):
        """Calculates the x-value of a given z-score."""
        return self.mean + (z * self.stddev)

    def pdf(self, x):
        """Calculates the PDF value for a given x-value.

        Args:
            x (float): x-value

        Returns:
            float: PDF value for x
        """
        pi = 3.1415926536
        e = 2.7182818285

        z = (x - self.mean) / self.stddev
        exponent = e ** (-0.5 * (z ** 2))
        denominator = self.stddev * ((2 * pi) ** 0.5)

        return (1 / denominator) * exponent

    def cdf(self, x):
        """Calculates the CDF value for a given x-value.

        Args:
            x (float): x-value

        Returns:
            float: CDF value for x
        """
        pi = 3.1415926536

        z = (x - self.mean) / (self.stddev * (2 ** 0.5))

        erf = (2 / (pi ** 0.5)) * (
            z
            - (z ** 3) / 3
            + (z ** 5) / 10
            - (z ** 7) / 42
            + (z ** 9) / 216
        )

        return 0.5 * (1 + erf)
