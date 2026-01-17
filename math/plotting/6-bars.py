#!/usr/bin/env python3
"""
This module contains a script that plots a stacked bar graph
representing fruit quantities per person.
"""
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(5)
fruit = np.random.randint(0, 20, (4, 3))

# Define the names and colors for clarity
people = ['Farrah', 'Fred', 'Felicia']
fruit_names = ['apples', 'bananas', 'oranges', 'peaches']
colors = ['red', 'yellow', '#ff8000', '#ffe5b4']

# Create an array for the x-positions [0, 1, 2]
x_pos = np.arange(len(people))

# Plot 1: Apples (The base)
plt.bar(x_pos, fruit[0], width=0.5, color=colors[0], label=fruit_names[0])

# Plot 2: Bananas (Stacked on top of Apples)
# 'bottom=fruit[0]' tells it to start drawing where the apples ended
plt.bar(x_pos, fruit[1], width=0.5, color=colors[1],
        label=fruit_names[1], bottom=fruit[0])

# Plot 3: Oranges (Stacked on top of Apples + Bananas)
plt.bar(x_pos, fruit[2], width=0.5, color=colors[2],
        label=fruit_names[2], bottom=fruit[0] + fruit[1])

# Plot 4: Peaches (Stacked on top of everything else)
plt.bar(x_pos, fruit[3], width=0.5, color=colors[3],
        label=fruit_names[3], bottom=fruit[0] + fruit[1] + fruit[2])

# Styling the axis and titles
plt.title('Number of Fruit per Person')
plt.ylabel('Quantity of Fruit')

# Set y-axis range (0 to 80) and ticks (every 10 units)
plt.yticks(np.arange(0, 81, 10))

# Label the x-axis ticks with the names of the people
plt.xticks(x_pos, people)

# Add the legend
plt.legend()

# Display the plot
plt.show()
