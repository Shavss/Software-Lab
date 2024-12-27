"""
@file metrics.py
@brief Module for computing evaluation metrics for model predictions.
    
This module provides functions to compute the Intersection over Union (IoU) and Dice Coefficient
for evaluating the performance of segmentation models.
"""

import numpy as np

def compute_iou(y_true, y_pred):
    """
    @brief Computes the Intersection over Union (IoU) between ground truth and prediction.
    
    This function calculates the IoU metric, which measures the overlap between the predicted
    segmentation and the ground truth.
    
    @param y_true (numpy.ndarray): Ground truth binary mask.
    @param y_pred (numpy.ndarray): Predicted binary mask.
    
    @return float: IoU score.
    """
    intersection = np.logical_and(y_true, y_pred).sum()
    union = np.logical_or(y_true, y_pred).sum()
    if union == 0:
        return 1.0  # If both masks are empty, define IoU as 1
    return intersection / union

def compute_dice(y_true, y_pred):
    """
    @brief Computes the Dice Coefficient between ground truth and prediction.
    
    This function calculates the Dice Coefficient, which measures the similarity between
    the predicted segmentation and the ground truth.
    
    @param y_true (numpy.ndarray): Ground truth binary mask.
    @param y_pred (numpy.ndarray): Predicted binary mask.
    
    @return float: Dice Coefficient score.
    """
    intersection = np.logical_and(y_true, y_pred).sum()
    sum_masks = y_true.sum() + y_pred.sum()
    if sum_masks == 0:
        return 1.0  # If both masks are empty, define Dice as 1
    return (2. * intersection) / sum_masks

def compute_metrics_over_validation(X_val, y_val, model):
    """
    @brief Computes IoU and Dice Coefficient over the entire validation set.
    
    This function generates predictions for the validation set using the trained model and
    computes the IoU and Dice Coefficient for each sample. It returns the mean IoU and Dice
    scores across all validation samples.
    
    @param X_val (numpy.ndarray): Validation input images.
    @param y_val (numpy.ndarray): Ground truth masks.
    @param model (tensorflow.keras.Model): Trained U-Net model.
    
    @return tuple: Mean IoU and Mean Dice Coefficient.
    """
    print("Generating predictions for the entire validation set...")
    pred_masks = model.predict(X_val)
    pred_masks = (pred_masks > 0.5).astype(np.float32)  # Thresholding to obtain binary masks

    ious = []
    dice_scores = []

    print("Calculating IoU and Dice Coefficient for each sample...")
    for i in range(len(X_val)):
        y_true_binary = y_val[i].squeeze().astype(np.float32)
        y_pred_binary = pred_masks[i].squeeze().astype(np.float32)
        
        # Ensure binary masks
        y_true_binary = (y_true_binary > 0.5).astype(np.float32)
        y_pred_binary = (y_pred_binary > 0.5).astype(np.float32)

        iou = compute_iou(y_true_binary, y_pred_binary)
        dice = compute_dice(y_true_binary, y_pred_binary)

        ious.append(iou)
        dice_scores.append(dice)

        if (i+1) % 500 == 0 or (i+1) == len(X_val):
            print(f"Processed {i+1}/{len(X_val)} samples.")

    mean_iou = np.mean(ious)
    mean_dice = np.mean(dice_scores)

    print(f"Mean IoU over the entire validation set: {mean_iou:.4f}")
    print(f"Mean Dice Coefficient over the entire validation set: {mean_dice:.4f}")

    return mean_iou, mean_dice
