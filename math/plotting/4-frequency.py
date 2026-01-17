#!/usr/bin/env python3
"""
This module contains a script that plots a histogram of student scores
with bins every 10 units.
"""
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(5)
student_grades = np.random.normal(68, 15, 50)

# Plot the histogram
# bins=range(0, 101, 10) creates bins [0, 10, 20... 100]
# edgecolor='black' adds the black outline to the bars
plt.hist(student_grades, bins=range(0, 101, 10), edgecolor='black')

# Set the title and axis labels
plt.title("Project A")
plt.xlabel("Grades")
plt.ylabel("Number of Students")

# Set the x-axis limits to match the bins (0 to 100)
plt.xlim(0, 100)

# Display the plot
plt.show()
