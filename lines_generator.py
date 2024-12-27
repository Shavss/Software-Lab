"""
@file line_drawing_generation.py
@brief Module for generating line drawings, converting between SVG, PDF, and PNG formats.

This module provides functionalities to generate SVG line drawings with crossing lines,
convert SVG files to PDF and PNG formats, and batch process multiple images with varying
numbers of lines.

@dependencies
- google.colab
- os
- svgwrite
- random
- svglib
- reportlab
- PIL
- IPython.display
- time
- pdf2image
- cairosvg
"""

from google.colab import drive
import os
import svgwrite
import random
from svglib.svglib import svg2rlg
from reportlab.graphics import renderPDF
from PIL import Image
from IPython.display import display, SVG
import time
from pdf2image import convert_from_path
import cairosvg

# Path to Google Drive directory
gdrive_path = '...'

# Mount Google Drive
drive.mount('/content/gdrive', force_remount=True)
os.chdir(gdrive_path)
print(sorted(os.listdir()))

def generate_crossing_lines_svg(file_name, num_lines, width=160, height=160):
    """
    @brief Generates an SVG file with a specified number of crossing lines.

    This function creates an SVG image containing a given number of randomly positioned
    lines within the specified width and height.

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

    This function takes an SVG file, renders it, and saves the output as a PDF file.

    @param svg_file_name (str): The name of the source SVG file.
    @param pdf_file_name (str): The name of the output PDF file.

    @return None
    """
    drawing = svg2rlg(svg_file_name)
    renderPDF.drawToFile(drawing, pdf_file_name)
    print(f"PDF saved as {pdf_file_name}")

def convert_svg_to_png(svg_file_name, png_file_name, width=160, height=160):
    """
    @brief Converts an SVG file to PNG format.

    This function reads an SVG file and converts it into a PNG image with specified dimensions.

    @param svg_file_name (str): The name of the source SVG file.
    @param png_file_name (str): The name of the output PNG file.
    @param width (int, optional): The width of the output PNG image in pixels. Default is 160.
    @param height (int, optional): The height of the output PNG image in pixels. Default is 160.

    @return None
    """
    with open(svg_file_name, 'r') as f:
        svg_content = f.read()
    cairosvg.svg2png(bytestring=svg_content, write_to=png_file_name, output_width=width, output_height=height)

    print(f"PNG saved as {png_file_name}")

def generate_multiple_images():
    """
    @brief Generates multiple SVG and PDF images with varying numbers of lines.

    This function creates a dataset of SVG and PDF images, each containing a specific number
    of crossing lines. It balances the number of samples across different line counts and
    stores them in the 'data_8000' directory.

    @return None
    """
    num_samples_per_line = 1000  # Number of samples per number of lines
    i = 0

    for num_lines in range(3, 9):
        line_count = 0
        while line_count < num_samples_per_line:
            svg_file_name = os.path.join('data_8000', f'000_{num_lines}_{line_count:03d}.svg')
            pdf_file_name = os.path.join('data_8000', f'000_{num_lines}_{line_count:03d}.pdf')

            # Generate SVG and PDF
            generate_crossing_lines_svg(svg_file_name, num_lines=num_lines, width=160, height=160)
            convert_svg_to_pdf(svg_file_name, pdf_file_name)

            line_count += 1

            # Display the first SVG file generated
            if num_lines == 3 and line_count == 1:
                display(SVG(filename=svg_file_name))

def convert_pdf_to_png(pdf_file_name, png_file_name):
    """
    @brief Converts a PDF file to PNG format.

    This function takes a PDF file, converts the first page to an image, and saves it as a PNG file.

    @param pdf_file_name (str): The name of the source PDF file.
    @param png_file_name (str): The name of the output PNG file.

    @return None
    """
    images = convert_from_path(pdf_file_name, 72)
    images[0].save(png_file_name, 'PNG')
    print(f"PNG saved as {png_file_name}")

# Execute the image generation process
generate_multiple_images()

# List all PDF files in 'data_8000' directory
files = [f for f in os.listdir('data_8000') if f.endswith('.pdf')]
i = 0
# Iterate through PDF files
for pdf_file in files:
    # Extract number of lines from PDF filename
    num_lines = int(pdf_file.split('_', 2)[1])  # Assumes filename format is '000_{num_lines}_{line_count:03d}.pdf'
    parts = pdf_file.replace('.pdf', '').split('_')
    # Extract the line_count
    line_count = int(parts[-1])
    
    # Construct paths and filename with number of lines
    pdf_file_path = os.path.join('data_8000', pdf_file)
    png_file_name = os.path.join('data_8000', f'000_{num_lines}_{line_count:03d}.png')

    # Convert PDF to PNG
    convert_pdf_to_png(pdf_file_path, png_file_name)

    # Increment index
    i += 1
