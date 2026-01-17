#!/usr/bin/env python3
"""
This module contains a script that plots two line graphs
representing exponential decay of C-14 and Ra-226.
"""
import numpy as np
import matplotlib.pyplot as plt

x = np.arange(0, 21000, 1000)
r = np.log(0.5)
t1 = 5730
t2 = 1600
y1 = np.exp((r / t1) * x)
y2 = np.exp((r / t2) * x)

# Plot C-14 (y1) as a dashed red line
plt.plot(x, y1, 'r--', label='C-14')

# Plot Ra-226 (y2) as a solid green line
plt.plot(x, y2, 'g-', label='Ra-226')

# Set the title and axis labels
plt.title("Exponential Decay of Radioactive Elements")
plt.xlabel("Time (years)")
plt.ylabel("Fraction Remaining")

# Set the axis limits
plt.xlim(0, 20000)
plt.ylim(0, 1)

# Add the legend to the upper right corner
plt.legend(loc='upper right')

# Display the plot
plt.show()
