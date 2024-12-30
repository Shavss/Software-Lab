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


from src.data_loading import count_files_with_extension, get_image_paths
from src.data_preprocessing import create_dataframe_from_svgs, group_and_pad_dataframe, count_line_counts
from src.utils import load_images_from_directory
from src.data_augmentation import get_data_generators
from src.training import compile_model, train_model, train_advanced_model, train_regression_model
from models import unet_model, create_advanced_model, create_enhanced_patch_transformer_model
from src.metrics import compute_metrics_over_validation
from src.post_processing import refine_and_save_as_svg, debug_detected_lines
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
 
    
    The user can choose which model to train: U-Net, Advanced Line Prediction, or Enhanced Patch Transformer.
    """
    # User selects the model to train
    print("Select the model to train:")
    print("1: U-Net")
    print("2: Advanced Line Prediction Model")
    print("3: Enhanced Patch Transformer Model")
    print("4: Load a pre-trained model for evaluation")
    choice = input("Enter the number corresponding to your choice: ")

    folder_path = 'data/raw'

    ## **Data Preparation**
    # Load input images and parse line data
    print("Loading data...")
    input_images, _ = load_images_from_directory(folder_path, extension=".png", normalize=True)
    df = create_dataframe_from_svgs(folder_path)
    target = group_and_pad_dataframe(df, max_lines=8)
    coords = np.array([entry[2] for entry in target])

    # Split data
    print("Splitting data into training and validation sets...")
    X_train, X_val, y_train, y_val = train_test_split(input_images, coords, test_size=0.2, random_state=42)

    # Generate masks for U-Net
    from post_processing import create_masks
    masks = create_masks(coords)
    masks = np.expand_dims(masks, axis=-1)

    # Split masks
    X_train_masks, X_val_masks, y_train_masks, y_val_masks = train_test_split(input_images, masks, test_size=0.2, random_state=42)

    # Generate combined labels for advanced models
    train_combined = np.concatenate([y_train, np.ones_like(y_train[:, :, :1])], axis=-1)
    val_combined = np.concatenate([y_val, np.ones_like(y_val[:, :, :1])], axis=-1)

    ## **Model Selection**
    if choice == "1":
        print("Training U-Net model...")
        model = unet_model(input_size=(160, 160, 1))
        model = compile_model(model, loss='binary_crossentropy', learning_rate=1e-4)

        train_gen, val_gen = get_data_generators(X_train_masks, y_train_masks, X_val_masks, y_val_masks, batch_size=16)
        history = train_model(model, train_gen, val_gen, X_train_masks, X_val_masks, batch_size=16, epochs=100, model_name="unet")

        # Post-process and save results
        pred_masks = model.predict(X_val_masks)
        pred_masks = (pred_masks > 0.5).astype(np.float32)
        refine_and_save_as_svg(pred_masks, svg_output_dir='results/unet_svgs/')

    elif choice == "2":
        print("Training Advanced Line Prediction Model...")
        model = create_advanced_model()
        history = train_advanced_model(X_train, train_combined, X_val, val_combined, batch_size=32, epochs=50)

        # Post-process and save results
        advanced_preds = model.predict(X_val)
        advanced_coords, _ = np.split(advanced_preds, [-1], axis=-1)
        refine_and_save_as_svg(advanced_coords, svg_output_dir='results/advanced_svgs/')

    elif choice == "3":
        print("Training Enhanced Patch Transformer Model...")
        model = create_enhanced_patch_transformer_model()
        model = compile_model(model, loss='binary_crossentropy', learning_rate=1e-4)

        history = train_model(model, None, None, X_train, X_val, batch_size=32, epochs=50, model_name="transformer")

        # Post-process and save results
        transformer_preds = model.predict(X_val)
        refine_and_save_as_svg(transformer_preds, svg_output_dir='results/transformer_svgs/')

    elif choice == "4":
        model_path = input("Enter the path to the saved model: ")
        print(f"Loading model from {model_path}...")
        model = load_model(model_path)

        # Evaluate and post-process
        print("Evaluating the model...")
        mean_iou, mean_dice = compute_metrics_over_validation(X_val, masks, model)
        print(f"Mean IoU: {mean_iou:.4f}")
        print(f"Mean Dice Coefficient: {mean_dice:.4f}")

        print("Post-processing predictions...")
        pred_masks = model.predict(X_val)
        pred_masks = (pred_masks > 0.5).astype(np.float32)
        refine_and_save_as_svg(pred_masks, svg_output_dir='results/evaluated_svgs/')

    else:
        print("Invalid choice. Exiting.")
        return

    ## **Plot Training History**
    if choice in ["1", "2", "3"]:
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.plot(history.history['loss'], label='Train Loss')
        plt.plot(history.history['val_loss'], label='Val Loss')
        plt.legend()
        plt.title('Loss')

        if 'accuracy' in history.history:
            plt.subplot(1, 2, 2)
            plt.plot(history.history['accuracy'], label='Train Accuracy')
            plt.plot(history.history['val_accuracy'], label='Val Accuracy')
            plt.legend()
            plt.title('Accuracy')

        plt.show()

if __name__ == "__main__":
    main()