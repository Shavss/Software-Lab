"""
@file visualization.py
@brief Module for visualizing images and their corresponding lines.

This module provides functions to display images and overlay line coordinates for visualization purposes.
"""

import matplotlib.pyplot as plt
from IPython.display import Image, display

def display_sample_image(image_path):
    """
    @brief Displays an image using IPython's display utilities.

    This function loads and displays an image from the specified path.

    @param image_path (str): Path to the image file.

    @return None
    """
    display(Image(filename=image_path))

def plot_lines_on_image(line_coords, img_size=160):
    """
    @brief Plots lines on a matplotlib figure.

    This function takes line coordinates, scales them according to image size,
    and plots them on a graph for visualization.

    @param line_coords (list): List of line coordinates, each as [x1, y1, x2, y2].
    @param img_size (int, optional): Size of the image to scale coordinates. Default is 160.

    @return None
    """
    plt.figure(figsize=(4, 4))
    for line in line_coords:
        if line == [0, 0, 0, 0]:
            continue

        x1, y1, x2, y2 = [coord * img_size for coord in line]
        plt.plot([x1, x2], [y1, y2], 'r-', linewidth=2)

    plt.xlim(0, img_size)
    plt.ylim(0, img_size)
    plt.gca().invert_yaxis()  # Invert y-axis to match image coordinates
    plt.axis('off')
    plt.show()
