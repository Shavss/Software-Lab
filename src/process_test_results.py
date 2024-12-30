"""
@file process_test_results.py
@brief Module for processing test images, generating predictions, and saving results.

This module loads test PNG images, applies a trained model to generate predictions, 
and uses the post-processing pipeline to refine and save results as SVG files.
"""

import os
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from post_processing import (
    post_process_mask,
    detect_lines_skeleton,
    merge_nearby_lines,
    refine_and_save_as_svg
)
import matplotlib.pyplot as plt


def load_test_images(test_folder, target_size=(160, 160)):
    """
    Load PNG images from a test folder, resize them, and normalize.

    @param test_folder (str): Path to the folder containing test PNG images.
    @param target_size (tuple): Target size to resize images. Default is (160, 160).

    @return tuple: 
        - numpy.ndarray: Array of preprocessed test images.
        - list: List of filenames corresponding to the images.
    """
    images = []
    filenames = []
    for filename in sorted(os.listdir(test_folder)):
        if filename.endswith('.png'):
            img_path = os.path.join(test_folder, filename)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, target_size)
            img = img.astype(np.float32) / 255.0
            images.append(np.expand_dims(img, axis=-1))  # Add channel dimension
            filenames.append(filename)
    return np.array(images), filenames


def process_test_folder(test_folder, model_path, svg_output_dir, target_size=(160, 160)):
    """
    Process PNG images from a test folder, predict masks, and save results as SVG.

    @param test_folder (str): Path to the folder containing PNG images.
    @param model_path (str): Path to the saved model.
    @param svg_output_dir (str): Directory to save the SVG results.
    @param target_size (tuple): Target size for resizing images. Default is (160, 160).

    @return None
    """
    # Load the trained model
    model = load_model(model_path)

    # Load test images
    test_images, filenames = load_test_images(test_folder, target_size)

    # Predict masks
    pred_masks = model.predict(test_images)
    pred_masks = (pred_masks > 0.5).astype(np.float32)

    # Use the post-processing module to refine and save predictions as SVG
    refine_and_save_as_svg(pred_masks, filenames, svg_output_dir, original_images=test_images)


def visualize_results(original_img, skeleton, mask, svg_path=None, title="Visualization"):
    """
    Visualize the original image, skeletonized image, mask, and optional SVG lines.

    @param original_img: The original input image (numpy array).
    @param skeleton: Skeletonized image (numpy array).
    @param mask: Predicted mask (numpy array).
    @param svg_path: Path to the SVG file for overlay visualization (optional).
    @param title: Title for the visualization plot.
    """
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    axes[0].imshow(original_img, cmap='gray')
    axes[0].set_title('Original Image')
    axes[0].axis('off')

    axes[1].imshow(skeleton, cmap='gray')
    axes[1].set_title('Skeletonized')
    axes[1].axis('off')

    axes[2].imshow(mask, cmap='gray')
    axes[2].set_title('Predicted Mask')
    axes[2].axis('off')

    if svg_path and os.path.exists(svg_path):
        import xml.etree.ElementTree as ET
        root = ET.parse(svg_path).getroot()
        axes[3].imshow(original_img, cmap='gray')
        axes[3].set_title('SVG Overlay')
        axes[3].axis('off')
        for line in root.findall('.//{http://www.w3.org/2000/svg}line'):
            x1, y1, x2, y2 = map(float, [line.attrib['x1'], line.attrib['y1'], line.attrib['x2'], line.attrib['y2']])
            axes[3].plot([x1, x2], [y1, y2], color='red')
    else:
        axes[3].imshow(original_img, cmap='gray')
        axes[3].set_title('No SVG File')
        axes[3].axis('off')

    fig.suptitle(title)
    plt.show()
