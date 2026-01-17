#!/usr/bin/env python3
"""
This module contains a script that plots 5 graphs in one figure
arranged in a 3x2 grid.
"""
import numpy as np
import matplotlib.pyplot as plt

# --- Data Generation (Provided) ---
y0 = np.arange(0, 11) ** 3

mean = [69, 0]
cov = [[15, 8], [8, 15]]
np.random.seed(5)
x1, y1 = np.random.multivariate_normal(mean, cov, 2000).T
y1 += 180

x2 = np.arange(0, 28651, 5730)
r2 = np.log(0.5)
t2 = 5730
y2 = np.exp((r2 / t2) * x2)

x3 = np.arange(0, 21000, 1000)
r3 = np.log(0.5)
t31 = 5730
t32 = 1600
y31 = np.exp((r3 / t31) * x3)
y32 = np.exp((r3 / t32) * x3)

np.random.seed(5)
student_grades = np.random.normal(68, 15, 50)

# --- Plotting Code ---

# Create the main figure and set the main title
fig = plt.figure()
fig.suptitle("All in One")

# 1. Top Left: y = x^3
plt.subplot(3, 2, 1)
plt.plot(y0, 'r-')
plt.xlim(0, 10)
plt.title('y = x^3', fontsize='x-small')
plt.xlabel('x', fontsize='x-small')
plt.ylabel('y', fontsize='x-small')

# 2. Top Right: Scatter Plot (Height vs Weight)
plt.subplot(3, 2, 2)
plt.scatter(x1, y1, c='magenta')
plt.title("Men's Height vs Weight", fontsize='x-small')
plt.xlabel("Height (in)", fontsize='x-small')
plt.ylabel("Weight (lbs)", fontsize='x-small')

# 3. Middle Left: Log Scale (C-14)
plt.subplot(3, 2, 3)
plt.plot(x2, y2)
plt.yscale('log')
plt.xlim(0, 28650)
plt.title("Exponential Decay of C-14", fontsize='x-small')
plt.xlabel("Time (years)", fontsize='x-small')
plt.ylabel("Fraction Remaining", fontsize='x-small')

# 4. Middle Right: Two Lines (Decay)
plt.subplot(3, 2, 4)
plt.plot(x3, y31, 'r--', label='C-14')
plt.plot(x3, y32, 'g-', label='Ra-226')
plt.xlim(0, 20000)
plt.ylim(0, 1)
plt.title("Exponential Decay of Radioactive Elements", fontsize='x-small')
plt.xlabel("Time (years)", fontsize='x-small')
plt.ylabel("Fraction Remaining", fontsize='x-small')
plt.legend(loc='upper right', fontsize='x-small')

# 5. Bottom: Histogram (Spanning 2 columns)
# Note: subplot(3, 1, 3) effectively means "Row 3, taking the whole width"
plt.subplot(3, 1, 3)
plt.hist(student_grades, bins=range(0, 101, 10), edgecolor='black')
plt.xlim(0, 100)
plt.ylim(0, 30)
