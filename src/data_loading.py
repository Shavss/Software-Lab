"""
@file data_loading.py
@brief Module for loading and counting data files.

This module provides functions to count files with specific extensions and to load image paths from directories.
"""

import os

def count_files_with_extension(folder_path, extension):
    """
    @brief Counts the number of files with a specific extension in a folder.

    This function iterates through the specified folder and counts how many files end with the given extension.

    @param folder_path (str): Path to the target folder.
    @param extension (str): File extension to count (e.g., '.svg', '.pdf', '.png').

    @return int: Number of files matching the extension.
    """
    count = 0
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(extension):
            count += 1
    return count

def get_image_paths(directory_path, extension=".png"):
    """
    @brief Retrieves sorted image file paths from a directory.

    This function collects and sorts file paths with the specified extension from the given directory.

    @param directory_path (str): Path to the directory containing image files.
    @param extension (str, optional): File extension to filter by. Default is ".png".

    @return list: Sorted list of image file paths.
    """
    return sorted(
        [
            os.path.join(directory_path, fname)
            for fname in os.listdir(directory_path)
            if fname.lower().endswith(extension)
        ]
    )
