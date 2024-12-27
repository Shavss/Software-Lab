"""
@file utils.py
@brief Utility functions for image processing.

This module provides utility functions to load images as matrices using OpenCV.
"""

import os
import cv2
import numpy as np

def load_image_as_matrix_cv2(file_path, normalize=True):
    """
    @brief Loads an image as a grayscale matrix using OpenCV.

    This function reads an image from the specified path, converts it to grayscale,
    normalizes it if required, and reshapes it to include a channel dimension.

    @param file_path (str): Path to the image file.
    @param normalize (bool, optional): Whether to normalize pixel values to [0, 1]. Default is True.

    @return numpy.ndarray: Image matrix with shape (height, width, 1).
    """
    image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Image not found at path: {file_path}")
    if normalize:
        image = image.astype(np.float32) / 255.0
    image = np.expand_dims(image, axis=-1)  # Shape becomes (height, width, 1)
    return image

def load_images_from_directory(path, extension=".png", normalize=True):
    """
    @brief Loads and sorts image matrices from a directory.

    This function retrieves image file paths with the specified extension from the given directory,
    sorts them, and loads each image as a matrix.

    @param path (str): Path to the directory containing image files.
    @param extension (str, optional): File extension to filter by. Default is ".png".
    @param normalize (bool, optional): Whether to normalize pixel values to [0, 1]. Default is True.

    @return tuple: Numpy array of image matrices and list of corresponding filenames.
    """
    def sort_key(filename):
        # Extract numerical parts from the filename
        parts = os.path.splitext(filename)[0].split('_')
        try:
            return tuple(map(int, parts))
        except ValueError:
            return (0,)

    # Filter and sort filenames
    filenames = sorted(
        [f for f in os.listdir(path) if f.lower().endswith(extension)],
        key=sort_key
    )

    image_matrices = []
    image_filenames = []
    for filename in filenames:
        print(f"Loading image: {filename}")
        file_path = os.path.join(path, filename)
        try:
            image_matrix = load_image_as_matrix_cv2(file_path, normalize=normalize)
            image_matrices.append(image_matrix)
            image_filenames.append(filename)
        except FileNotFoundError as e:
            print(e)
    return np.array(image_matrices), image_filenames
