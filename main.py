"""
@file main.py
@brief Main script for data exploration, preparation, model training, and evaluation.
    
This script orchestrates the loading, preprocessing, training, evaluation, and visualization of SVG and image data
for deep learning applications using a U-Net model.
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from data_loading import count_files_with_extension, get_image_paths
from data_preprocessing import (
    parse_svg,
    create_dataframe_from_svgs,
    group_and_pad_dataframe,
    count_line_counts
)
from utils import load_images_from_directory
from visualization import display_sample_image, plot_lines_on_image
from model import unet_model
from data_augmentation import get_data_generators
from training import compile_model, train_model
from post_processing import refine_and_save_as_svg, debug_detected_lines
from metrics import compute_metrics_over_validation
from tensorflow.keras.models import load_model

def main():
    """
    @brief Executes the data exploration, preparation, model training, and evaluation workflow.
    
    This function performs the following steps:
    1. Counts SVG, PDF, and PNG files in the data directory.
    2. Parses SVG files and creates a DataFrame.
    3. Groups and pads the DataFrame for consistency.
    4. Counts the number of files per line count.
    5. Prepares paths for input images and target SVGs.
    6. Loads images as matrices.
    7. Splits data into training and validation sets.
    8. Creates data generators for augmentation.
    9. Defines and compiles the U-Net model.
    10. Trains the model with early stopping and checkpointing.
    11. Plots training history.
    12. Computes evaluation metrics over the validation set.
    13. Post-processes model predictions and saves them as SVGs.
    14. Visualizes sample predictions.
    """
    ## **START HERE: Data Exploration and Preparation**
    
    # Count the number of SVG, PDF, and PNG files
    folder_path = 'data_8000'
    svg_count = count_files_with_extension(folder_path, '.svg')
    pdf_count = count_files_with_extension(folder_path, '.pdf')
    png_count = count_files_with_extension(folder_path, '.png')
    
    print(f"Number of SVG files: {svg_count}")
    print(f"Number of PDF files: {pdf_count}")
    print(f"Number of PNG files: {png_count}")
    
    ## **Parsing SVGs with Already Normalized Coordinate Values and Creating DataFrame**
    
    # Create a DataFrame from parsed SVG files
    df = create_dataframe_from_svgs(folder_path)
    print(df.head())
    
    ## ***GROUPING AND PADDING the DataFrame***
    
    target = group_and_pad_dataframe(df, max_lines=8)
    
    for entry in target[:10]:
        print(entry)
    
    line_count_dict = count_line_counts(target)
    
    for num_lines, count in line_count_dict.items():
        print(f"Number of files with {num_lines} lines: {count}")
    
    ## **Prepare Paths of Input Images and Target SVGs**
    input_dir = "data_8000"
    target_dir = "data_8000/"
    img_size = (160, 160)
    input_img_paths = get_image_paths(input_dir, extension=".png")
    
    target_img_paths = sorted(
        [
            os.path.join(target_dir, fname)
            for fname in os.listdir(target_dir)
            if fname.endswith(".svg") and not fname.startswith(".")
        ]
    )
    
    print("Number of samples:", len(input_img_paths))
    
    for input_path, target_path in zip(input_img_paths[:10], target_img_paths[:10]):
        print(input_path, "|", target_path)
    
    ## **Load Images as Matrices**
    
    input_images, image_filenames = load_images_from_directory(folder_path, extension=".png", normalize=True)
    
    # Extract line counts and coordinates
    line_counts = np.array([entry[1] for entry in target])
    coords = np.array([entry[2] for entry in target])
    
    print(line_counts.shape)
    print(input_images.shape)
    print(coords.shape)
    
    ## **Split Data into Training and Validation Sets**
    X_train, X_val, y_train, y_val = train_test_split(input_images, coords, test_size=0.2, random_state=42)
    
    # Create target masks from coordinates
    from post_processing import create_masks
    masks = create_masks(coords)
    masks = np.expand_dims(masks, axis=-1)  # Shape: (num_samples, 160, 160, 1)
    
    # Split masks into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(input_images, masks, test_size=0.2, random_state=42)
    
    print(f"Training samples: {X_train.shape[0]}")
    print(f"Validation samples: {X_val.shape[0]}")
    
    ## **Define and Compile the U-Net Model**
    model = unet_model()
    model = compile_model(model)
    
    ## **Create Data Generators for Augmentation**
    train_gen, val_gen = get_data_generators(X_train, y_train, X_val, y_val, batch_size=16)
    
    ## **Train the Model**
    history = train_model(model, train_gen, val_gen, X_train, X_val, batch_size=16, epochs=100)
    
    ## **Plot Training History**
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.legend()
    plt.title('Loss')
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Train Acc')
    plt.plot(history.history['val_accuracy'], label='Val Acc')
    plt.legend()
    plt.title('Accuracy')
    
    plt.show()
    
    ## **Load Trained U-Net Model for Prediction**
    model_path = 'models/unet_best_model.keras'  # Path to your saved model
    model = load_model(model_path)
    
    # Check the model architecture (optional)
    model.summary()
    
    ## **Post-Processing and Evaluation**
    
    from post_processing import post_process_mask, detect_lines_skeleton, merge_nearby_lines, refine_and_save_as_svg, debug_detected_lines
    from metrics import compute_metrics_over_validation
    
    # Compute evaluation metrics
    mean_iou, mean_dice = compute_metrics_over_validation(X_val, y_val, model)
    
    ## **Post-Processing: Skeletonization and Morphological Cleaning**
    
    def process_and_visualize(X_val, y_val, model, num_samples=20):
        """
        @brief Processes validation data, makes predictions, post-processes masks, and visualizes results.
    
        This function predicts masks for validation images, refines them, saves as SVGs,
        and visualizes sample predictions alongside ground truth masks.
    
        @param X_val (numpy.ndarray): Validation images.
        @param y_val (numpy.ndarray): Validation masks.
        @param model (tensorflow.keras.Model): Trained U-Net model.
        @param num_samples (int, optional): Number of samples to process and visualize. Default is 20.
    
        @return None
        """
        # Predict on the first `num_samples` from X_val
        pred_masks = model.predict(X_val[:num_samples])
        pred_masks = (pred_masks > 0.5).astype(np.float32)
    
        # Refine and save as SVG
        refine_and_save_as_svg(pred_masks)
    
        # Visualization and debugging
        for i in range(num_samples):
            print(f"Processing sample {i}...")
    
            # Post-process predicted mask
            skeleton = post_process_mask(pred_masks[i])
            lines = detect_lines_skeleton(skeleton)
            filtered_lines = merge_nearby_lines(lines)
    
            # Debug detected lines
            debug_detected_lines(skeleton, filtered_lines)
    
            # Compare input, ground truth, and predicted masks
            plt.figure(figsize=(12, 4))
    
            # Input image
            plt.subplot(1, 3, 1)
            plt.imshow(X_val[i].squeeze(), cmap='gray')
            plt.title('Input Image')
            plt.axis('off')
    
            # Ground truth mask
            plt.subplot(1, 3, 2)
            plt.imshow(y_val[i].squeeze(), cmap='gray')
            plt.title('Ground Truth Mask')
            plt.axis('off')
    
            # Predicted mask
            plt.subplot(1, 3, 3)
            plt.imshow(pred_masks[i].squeeze(), cmap='gray')
            plt.title('Predicted Mask')
            plt.axis('off')
    
            plt.show()
    
    # **Run the Pipeline**
    process_and_visualize(X_val, y_val, model, num_samples=20)

if __name__ == "__main__":
    main()
