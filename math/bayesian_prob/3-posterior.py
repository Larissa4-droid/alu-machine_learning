#!/usr/bin/env python3
"""
Contains a function that calculates the posterior probability
for various hypothetical probabilities given observed data.
"""
import numpy as np


def posterior(x, n, P, Pr):
    """
    Calculates the posterior probability for the various hypothetical
    probabilities of developing severe side effects given the data.

    Args:
        x: number of patients that develop severe side effects
        n: total number of patients observed
        P: 1D numpy.ndarray of hypothetical probabilities
        Pr: 1D numpy.ndarray of prior beliefs of P

    Returns:
        1D numpy.ndarray containing the posterior probability for each p in P
    """
    if not isinstance(n, int) or n <= 0:
        raise ValueError("n must be a positive integer")

    if not isinstance(x, int) or x < 0:
        raise ValueError(
            "x must be an integer that is greater than or equal to 0"
        )

    if x > n:
        raise ValueError("x cannot be greater than n")

    if not isinstance(P, np.ndarray) or len(P.shape) != 1:
        raise TypeError("P must be a 1D numpy.ndarray")

    if not isinstance(Pr, np.ndarray) or Pr.shape != P.shape:
        raise TypeError("Pr must be a numpy.ndarray with the same shape as P")

    if np.any((P < 0) | (P > 1)):
        raise ValueError("All values in P must be in the range [0, 1]")

    if np.any((Pr < 0) | (Pr > 1)):
        raise ValueError("All values in Pr must be in the range [0, 1]")

    if not np.isclose(np.sum(Pr), 1):
        raise ValueError("Pr must sum to 1")

    # 1. Calculate Likelihood (Binomial Distribution)
    fact = np.math.factorial
    combination = fact(n) / (fact(x) * fact(n - x))
    likelihood = combination * (P ** x) * ((1 - P) ** (n - x))

    # 2. Calculate Intersection (Likelihood * Prior)
    intersection = likelihood * Pr

    # 3. Calculate Marginal (Sum of Intersections)
    marginal = np.sum(intersection)

    # 4. Calculate Posterior (Intersection / Marginal)
    post = intersection / marginal

    return post
