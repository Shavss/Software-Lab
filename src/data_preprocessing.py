"""
@file data_preprocessing.py
@brief Module for parsing SVG files and preparing data structures.

This module provides functions to parse SVG files, create a pandas DataFrame from parsed data,
and group and pad the DataFrame for further processing.
"""

import os
import xml.etree.ElementTree as ET
import pandas as pd

def parse_svg(file_content):
    """
    @brief Parses SVG content to extract line data.

    This function parses the SVG content, extracts line coordinates, normalizes them,
    and returns the number of lines and their details.

    @param file_content (str): Content of the SVG file as a string.

    @return tuple: Number of lines (int) and a list of dictionaries containing line data.
    """
    root = ET.fromstring(file_content)
    ns = {'svg': 'http://www.w3.org/2000/svg'}
    lines = root.findall('.//svg:line', namespaces=ns)

    line_data = []
    for line in lines:
        x1 = float(line.get('x1')) / 160
        y1 = float(line.get('y1')) / 160
        x2 = float(line.get('x2')) / 160
        y2 = float(line.get('y2')) / 160
        stroke_width = line.get('stroke-width')
        line_data.append({'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2, 'stroke-width': stroke_width})

    num_lines = len(lines)
    return num_lines, line_data

def create_dataframe_from_svgs(data_folder):
    """
    @brief Creates a pandas DataFrame from SVG files in a directory.

    This function iterates through all SVG files in the specified directory, parses each file,
    and constructs a DataFrame containing file names, number of lines, line coordinates, and stroke widths.

    @param data_folder (str): Path to the folder containing SVG files.

    @return pd.DataFrame: DataFrame with parsed SVG data.
    """
    data = []

    for filename in os.listdir(data_folder):
        if filename.endswith('.svg'):
            file_path = os.path.join(data_folder, filename)
            with open(file_path, 'r') as file:
                file_content = file.read()
            num_lines, line_data = parse_svg(file_content)
            for i, line in enumerate(line_data):
                data.append({
                    'file_name': filename,
                    'number_of_lines': num_lines,
                    'line_number': i + 1,
                    'x1': line["x1"],
                    'y1': line["y1"],
                    'x2': line["x2"],
                    'y2': line["y2"],
                    'stroke-width': line["stroke-width"]
                })

    df = pd.DataFrame(data)
    return df

def group_and_pad_dataframe(df, max_lines=8):
    """
    @brief Groups the DataFrame by file name and pads the line data.

    This function groups the DataFrame entries by file name, pads line data with zeros
    if the number of lines is less than the maximum required, and truncates if necessary.

    @param df (pd.DataFrame): DataFrame containing SVG line data.
    @param max_lines (int, optional): Maximum number of lines to retain per file. Default is 8.

    @return list: List of dictionaries containing grouped and padded line data.
    """
    target = []
    padding_value = [0, 0, 0, 0]
    for file_name, group in df.groupby('file_name'):
        number_of_lines = group['number_of_lines'].iloc[0]
        lines = group[['x1', 'y1', 'x2', 'y2']].values.tolist()

        if len(lines) < max_lines:
            lines += [padding_value] * (max_lines - len(lines))
        else:
            lines = lines[:max_lines]

        target.append([file_name, number_of_lines, lines])

    return target

def count_line_counts(target):
    """
    @brief Counts the number of files for each line count.

    This function creates a dictionary mapping the number of lines to the count of files
    having that many lines.

    @param target (list): List of dictionaries containing grouped and padded line data.

    @return dict: Dictionary with number of lines as keys and file counts as values.
    """
    line_count_dict = {}

    for entry in target:
        num_lines = entry[1]
        if num_lines in line_count_dict:
            line_count_dict[num_lines] += 1
        else:
            line_count_dict[num_lines] = 1

    return line_count_dict
