"""
@file line_drawing_generation.py
@brief Module for generating line drawings, converting between SVG, PDF, and PNG formats.

This module provides functionalities to generate SVG line drawings with crossing lines,
convert SVG files to PDF and PNG formats, and batch process multiple images with varying
numbers of lines, all within a local directory.
"""

import os
import svgwrite
import random
from svglib.svglib import svg2rlg
from reportlab.graphics import renderPDF
from PIL import Image
from matplotlib.pyplot import imshow
from pdf2image import convert_from_path
from IPython.display import SVG, display
import cairosvg


def generate_crossing_lines_svg(file_name, num_lines, width=160, height=160):
    """
    @brief Generates an SVG file with a specified number of crossing lines.

    @param file_name (str): The name of the SVG file to be saved.
    @param num_lines (int): The number of crossing lines to generate in the SVG.
    @param width (int, optional): The width of the SVG canvas in pixels. Default is 160.
    @param height (int, optional): The height of the SVG canvas in pixels. Default is 160.

    @return None
    """
    dwg = svgwrite.Drawing(file_name, size=(width, height))

    possible_stroke_widths = [1]

    for _ in range(num_lines):
        x1, y1 = random.randint(0, width), random.randint(0, height)
        x2, y2 = random.randint(0, width), random.randint(0, height)
        stroke_width = random.choice(possible_stroke_widths)
        dwg.add(dwg.line(start=(x1, y1), end=(x2, y2), stroke='black', stroke_width=stroke_width))

    dwg.save()
    print(f"SVG saved as {file_name}")


def convert_svg_to_pdf(svg_file_name, pdf_file_name):
    """
    @brief Converts an SVG file to PDF format.

    @param svg_file_name (str): The name of the source SVG file.
    @param pdf_file_name (str): The name of the output PDF file.

    @return None
    """
    drawing = svg2rlg(svg_file_name)
    renderPDF.drawToFile(drawing, pdf_file_name)
    print(f"PDF saved as {pdf_file_name}")


def convert_pdf_to_png(pdf_file_name, png_file_name, width=160, height=160):
    """
    @brief Converts a PDF file to PNG format.

    @param pdf_file_name (str): The name of the source PDF file.
    @param png_file_name (str): The name of the output PNG file.
    @param width (int, optional): The width of the output PNG image in pixels. Default is 160.
    @param height (int, optional): The height of the output PNG image in pixels. Default is 160.

    @return None
    """
    images = convert_from_path(pdf_file_name, dpi=72)
    if images:
        image = images[0].resize((width, height))
        image.save(png_file_name, 'PNG')
        print(f"PNG saved as {png_file_name}")
    else:
        print(f"No images found in PDF {pdf_file_name}")


def generate_multiple_images(data_folder='data', num_samples_per_line=100, min_lines=3, max_lines=8, width=160, height=160):
    """
    @brief Generates multiple SVG, PDF, and PNG images with varying numbers of lines.

    @param data_folder (str, optional): Directory to save generated SVG, PDF, and PNG files. Default is 'data'.
    @param num_samples_per_line (int, optional): Number of samples per number of lines. Default is 100.
    @param min_lines (int, optional): Minimum number of lines in SVG images. Default is 3.
    @param max_lines (int, optional): Maximum number of lines in SVG images. Default is 8.
    @param width (int, optional): Width of the SVG and PNG images in pixels. Default is 160.
    @param height (int, optional): Height of the SVG and PNG images in pixels. Default is 160.

    @return None
    """
    if not os.path.exists(data_folder):
        os.makedirs(data_folder)
        print(f"Created directory {data_folder}")

    for num_lines in range(min_lines, max_lines + 1):
        line_count = 0
        while line_count < num_samples_per_line:
            svg_file_name = os.path.join(data_folder, f'000_{num_lines}_{line_count:04d}.svg')
            pdf_file_name = os.path.join(data_folder, f'000_{num_lines}_{line_count:04d}.pdf')
            png_file_name = os.path.join(data_folder, f'000_{num_lines}_{line_count:04d}.png')

            # Generate SVG
            generate_crossing_lines_svg(svg_file_name, num_lines=num_lines, width=width, height=height)

            # Convert SVG to PDF
            convert_svg_to_pdf(svg_file_name, pdf_file_name)

            # Convert PDF to PNG
            convert_pdf_to_png(pdf_file_name, png_file_name, width=width, height=height)

            line_count += 1

            # Display the first SVG file generated for 3 lines as an example
            if num_lines == min_lines and line_count == 1:
                display(SVG(filename=svg_file_name))


if __name__ == "__main__":
    # Execute the image generation process
    generate_multiple_images(data_folder='data', num_samples_per_line=10)
