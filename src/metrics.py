"""
@file metrics.py
@brief Module for computing evaluation metrics for model predictions.
    
This module provides functions to compute the Intersection over Union (IoU), Dice Coefficient,
and additional metrics for evaluating segmentation models. It supports both binary masks and
models that include confidence scores.
"""

import numpy as np
import tensorflow as tf

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

def compute_advanced_metrics(y_true, y_pred, threshold=0.5):
    """
    @brief Computes metrics for Advanced models with confidence scores.
    
    This function calculates the IoU and Dice Coefficient for predicted line coordinates
    while accounting for confidence scores. Predictions below the confidence threshold
    are treated as absent lines.
    
    @param y_true (numpy.ndarray): Ground truth tensor of shape (batch_size, 8, 5).
                                   Last dimension contains (x1, y1, x2, y2, confidence).
    @param y_pred (numpy.ndarray): Predicted tensor of shape (batch_size, 8, 5).
    @param threshold (float, optional): Confidence threshold for predictions. Default is 0.5.
    
    @return tuple: Mean IoU and Mean Dice Coefficient for all samples.
    """
    pred_coords = y_pred[:, :, :4]
    pred_conf = y_pred[:, :, 4:]
    true_coords = y_true[:, :, :4]
    true_conf = y_true[:, :, 4:]

    ious = []
    dice_scores = []

    for i in range(y_true.shape[0]):  # Iterate over batch
        pred_coords_thresh = pred_coords[i][pred_conf[i].squeeze() > threshold]
        true_coords_thresh = true_coords[i][true_conf[i].squeeze() > threshold]

        if len(pred_coords_thresh) == 0 and len(true_coords_thresh) == 0:
            ious.append(1.0)
            dice_scores.append(1.0)
            continue

        pred_mask = np.zeros((160, 160))
        true_mask = np.zeros((160, 160))

        for line in pred_coords_thresh:
            x1, y1, x2, y2 = (line * 160).astype(int)
            pred_mask = cv2.line(pred_mask, (x1, y1), (x2, y2), 1, thickness=2)

        for line in true_coords_thresh:
            x1, y1, x2, y2 = (line * 160).astype(int)
            true_mask = cv2.line(true_mask, (x1, y1), (x2, y2), 1, thickness=2)

        ious.append(compute_iou(true_mask, pred_mask))
        dice_scores.append(compute_dice(true_mask, pred_mask))

    mean_iou = np.mean(ious)
    mean_dice = np.mean(dice_scores)

    return mean_iou, mean_dice


def compute_metrics_over_validation(X_val, y_val, model, advanced=False, threshold=0.5):
    """
    @brief Computes IoU and Dice Coefficient over the entire validation set.
    
    This function generates predictions for the validation set using the trained model and
    computes the IoU and Dice Coefficient for each sample. For advanced models, it also
    considers confidence scores in the evaluation.
    
    @param X_val (numpy.ndarray): Validation input images.
    @param y_val (numpy.ndarray): Ground truth masks or labels (coordinates + confidence scores).
    @param model (tensorflow.keras.Model): Trained model (U-Net or Advanced).
    @param advanced (bool, optional): Whether the model is advanced (predicting confidence scores). Default is False.
    @param threshold (float, optional): Confidence threshold for advanced models. Default is 0.5.
    
    @return tuple: Mean IoU and Mean Dice Coefficient.
    """
    print("Generating predictions for the entire validation set...")
    pred = model.predict(X_val)

    if advanced:
        print("Evaluating metrics for Advanced Model with confidence scores...")
        mean_iou, mean_dice = compute_advanced_metrics(y_val, pred, threshold)
    else:
        print("Evaluating metrics for U-Net Model...")
        pred_masks = (pred > 0.5).astype(np.float32)  # Threshold for binary masks
        ious = []
        dice_scores = []

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

        mean_iou = np.mean(ious)
        mean_dice = np.mean(dice_scores)

    print(f"Mean IoU over the validation set: {mean_iou:.4f}")
    print(f"Mean Dice Coefficient over the validation set: {mean_dice:.4f}")

    return mean_iou, mean_dice
