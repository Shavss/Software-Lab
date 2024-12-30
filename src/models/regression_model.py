"""
@file regression_model.py
@brief Defines a simple convolutional regression model for numerical prediction tasks.

@details
This module includes:
- A regression model for predicting a single numerical value from an input image.
- Training, evaluation, and visualization of the model's performance.
"""

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import matplotlib.pyplot as plt


def create_regression_model(input_shape=(160, 160, 1)):
    """
    @brief Creates and compiles a regression model using convolutional layers.

    @param input_shape (tuple, optional): Shape of the input images. Default is (160, 160, 1).
    
    @return Model: Compiled Keras regression model.
    """
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(64, activation='relu'),
        Dense(1)  # Output layer for regression
    ])

    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
    return model


def train_regression_model(model, train_images, train_labels, val_images, val_labels, epochs=13, batch_size=32):
    """
    @brief Trains the regression model and visualizes its performance.

    @param model (Model): Compiled Keras regression model.
    @param train_images (numpy.ndarray): Training image dataset.
    @param train_labels (numpy.ndarray): Labels corresponding to the training images.
    @param val_images (numpy.ndarray): Validation image dataset.
    @param val_labels (numpy.ndarray): Labels corresponding to the validation images.
    @param epochs (int, optional): Number of training epochs. Default is 13.
    @param batch_size (int, optional): Size of training batches. Default is 32.

    @return History: Keras History object containing training history.
    """
    history = model.fit(
        train_images, train_labels,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(val_images, val_labels)
    )
    return history


def evaluate_regression_model(model, test_images, test_labels):
    """
    @brief Evaluates the regression model on the test set.

    @param model (Model): Trained Keras regression model.
    @param test_images (numpy.ndarray): Test image dataset.
    @param test_labels (numpy.ndarray): Labels corresponding to the test images.

    @return tuple: Test loss and mean absolute error (MAE).
    """
    test_loss, test_mae = model.evaluate(test_images, test_labels)
    print(f"Test Loss (MSE): {test_loss:.2f}")
    print(f"Test MAE: {test_mae:.2f}")
    return test_loss, test_mae


def plot_training_history(history):
    """
    @brief Plots the training and validation loss over epochs.

    @param history (History): Keras History object containing training history.
    """
    plt.figure(figsize=(10, 5))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Mean Squared Error')
    plt.title('Training and Validation Loss Over Epochs')
    plt.legend()
    plt.show()


def predict_and_compare(model, test_images, test_labels, num_samples=10):
    """
    @brief Predicts on the test set and compares predictions with actual labels.

    @param model (Model): Trained Keras regression model.
    @param test_images (numpy.ndarray): Test image dataset.
    @param test_labels (numpy.ndarray): Labels corresponding to the test images.
    @param num_samples (int, optional): Number of predictions to display. Default is 10.
    """
    predictions = model.predict(test_images)
    for i in range(num_samples):
        print(f"Predicted: {predictions[i][0]:.2f}, Actual: {test_labels[i]}")
