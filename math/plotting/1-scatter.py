#!/usr/bin/env python3
"""
This module contains a script that plots a scatter graph
of Men's Height vs Weight.
"""
import numpy as np
import matplotlib.pyplot as plt

mean = [69, 0]
cov = [[15, 8], [8, 15]]
np.random.seed(5)
x, y = np.random.multivariate_normal(mean, cov, 2000).T
y += 180

# Plot x vs y as a scatter plot
# c='magenta' sets the color
plt.scatter(x, y, c='magenta')

# Set the title and axis labels
plt.title("Men's Height vs Weight")
plt.xlabel("Height (in)")
plt.ylabel("Weight (lbs)")

# Display the plot
plt.show()
