"""
@file advanced_model.py
@brief Defines the Advanced Line Prediction Model and its associated functions.

This module includes:
- Model definition with convolutional layers and confidence score prediction.
- Custom loss function for line coordinates and confidence scores.
- Utility functions for prediction and visualization.

@details
The Advanced Line Prediction Model predicts line coordinates and confidence scores
for each line in a given image. Confidence scores indicate the likelihood of line presence.
"""

import tensorflow as tf
from tensorflow.keras.layers import (Input, Conv2D, MaxPooling2D, Flatten, Dense, Reshape,
                                      GlobalAveragePooling2D, BatchNormalization, Dropout, Concatenate)
from tensorflow.keras.models import Model
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


def create_advanced_model():
    """
    @brief Creates and returns the Advanced Line Prediction Model.

    @return tensorflow.keras.Model: Compiled model.
    """
    inputs = Input(shape=(160, 160, 1))

    # Feature extraction layers
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)

    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)

    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)

    x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = GlobalAveragePooling2D()(x)

    # Fully connected layers
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(128, activation='relu')(x)

    # Output layers
    line_coordinates = Dense(8 * 4, activation='linear')(x)
    line_coordinates = Reshape((8, 4))(line_coordinates)
    confidence_scores = Dense(8, activation='sigmoid')(x)
    confidence_scores = Reshape((8, 1))(confidence_scores)

    outputs = Concatenate(axis=-1)([line_coordinates, confidence_scores])

    return Model(inputs, outputs, name="Advanced_Line_Prediction_Model")


def advanced_loss(y_true, y_pred):
    """
    @brief Computes the loss function for the model.

    @param y_true: Ground truth labels, including coordinates and confidence scores.
    @param y_pred: Predicted coordinates and confidence scores from the model.
    @return: Computed loss (float).
    """
    pred_coordinates = y_pred[:, :, :4]
    pred_confidence = y_pred[:, :, 4:]
    true_coordinates = y_true[:, :, :4]
    true_confidence = y_true[:, :, 4:]

    coord_loss = tf.reduce_mean(
        tf.reduce_sum(true_confidence * tf.square(pred_coordinates - true_coordinates), axis=-1)
    )
    confidence_loss = tf.reduce_mean(
        tf.keras.losses.binary_crossentropy(true_confidence, pred_confidence)
    )

    return coord_loss + confidence_loss


def apply_threshold(pred_coords, confidence_scores, threshold=0.5):
    """
    @brief Applies a threshold to predicted coordinates based on confidence scores.

    @param pred_coords: Predicted line coordinates.
    @param confidence_scores: Confidence scores for the predicted lines.
    @param threshold: Confidence threshold for retaining lines.
    @return: Thresholded coordinates with low-confidence lines set to zero.
    """
    for i in range(pred_coords.shape[0]):
        for j in range(pred_coords.shape[1]):
            if confidence_scores[i, j] < threshold:
                pred_coords[i, j, :] = 0
    return pred_coords


def predict_and_apply_threshold(model, test_images, threshold=0.5):
    """
    @brief Predicts and applies a threshold to the model's predictions.

    @param model: The trained model.
    @param test_images: Test dataset images.
    @param threshold: Confidence threshold for retaining predictions.
    @return: Thresholded predicted coordinates and confidence scores.
    """
    predictions = model.predict(test_images)
    pred_coords = predictions[:, :, :4]
    confidence_scores = predictions[:, :, 4:]
    pred_coords = apply_threshold(pred_coords, confidence_scores, threshold)
    return pred_coords, confidence_scores


def plot_lines(image, ground_truth_coords, predicted_coords, confidence_scores, threshold=0.5):
    """
    @brief Plots ground truth and predicted lines on an image.

    @param image: The input image (grayscale, 160x160).
    @param ground_truth_coords: Ground truth line coordinates.
    @param predicted_coords: Predicted line coordinates.
    @param confidence_scores: Predicted confidence scores.
    @param threshold: Confidence threshold for plotting predicted lines.
    """
    plt.figure(figsize=(8, 8))
    plt.imshow(image.squeeze(), cmap='gray')

    # Plot ground truth lines
    for coord in ground_truth_coords:
        plt.plot([coord[0]*160, coord[2]*160], [coord[1]*160, coord[3]*160], 'g-', label="Ground Truth")

    # Plot predicted lines
    for i, (coord, conf) in enumerate(zip(predicted_coords, confidence_scores)):
        if conf >= threshold:
            plt.plot([coord[0]*160, coord[2]*160], [coord[1]*160, coord[3]*160], 'r-', label="Predicted" if i == 0 else "")

    plt.legend()
    plt.title("Ground Truth vs Predicted Lines")
    plt.show()


# Example training and evaluation
# Ensure the data (input_images, labels) is loaded and preprocessed
input_images = np.random.rand(3000, 160, 160, 1)  # Placeholder for input images
coords = np.random.rand(3000, 8, 4)               # Placeholder for coordinates
confidence_scores = np.random.rand(3000, 8, 1)    # Placeholder for confidence scores
labels = np.concatenate([coords, confidence_scores], axis=-1)

X_train, X_val, y_train, y_val = train_test_split(input_images, labels, test_size=0.2, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_val, y_val, test_size=0.25, random_state=42)

advanced_model = create_advanced_model()
advanced_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
    loss=advanced_loss,
    metrics=['mean_squared_error']
)

history = advanced_model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    batch_size=32,
    epochs=50,
    callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=30)]
)

# Example prediction and visualization
single_image = X_test[5]
single_image_batch = np.expand_dims(single_image, axis=0)
pred_coords, confidence_scores = predict_and_apply_threshold(advanced_model, single_image_batch, threshold=0.8)
ground_truth_coords = y_test[5][:, :4]

plot_lines(single_image, ground_truth_coords, pred_coords[0], confidence_scores[0], threshold=0.8)

# Plot training history
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['mean_squared_error'], label='Training MSE')
plt.plot(history.history['val_mean_squared_error'], label='Validation MSE')
plt.title('Mean Squared Error')
plt.xlabel('Epochs')
plt.ylabel('MSE')
plt.legend()

plt.tight_layout()
plt.show()
