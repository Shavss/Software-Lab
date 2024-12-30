"""
@file training.py
@brief Module for compiling and training models.

This module provides functions to compile models, set up callbacks, and train them using
the provided data generators. It supports multiple architectures, including U-Net, Advanced Line Prediction Model,
Regression Model, and the Enhanced Patch Transformer Model.
"""

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from models import unet_model, create_advanced_model, create_regression_model, create_enhanced_patch_transformer_model, advanced_loss, scaled_loss


def compile_model(model, loss='binary_crossentropy', learning_rate=1e-4):
    """
    @brief Compiles a Keras model with the specified optimizer and loss function.

    This function configures the model for training by setting the optimizer, loss, and evaluation metrics.

    @param model (Model): Keras model to compile.
    @param loss (str or function, optional): Loss function to use. Default is 'binary_crossentropy'.
    @param learning_rate (float, optional): Learning rate for the optimizer. Default is 1e-4.

    @return Model: Compiled Keras model.
    """
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss=loss,
        metrics=['accuracy']
    )
    return model


def get_callbacks(model_checkpoint_path='models/best_model.keras'):
    """
    @brief Creates callbacks for training.

    This function sets up EarlyStopping to halt training when validation loss stops improving
    and ModelCheckpoint to save the best model during training.

    @param model_checkpoint_path (str, optional): Path to save the best model. Default is 'models/best_model.keras'.

    @return list: List of callback instances.
    """
    earlystopper = EarlyStopping(
        monitor='val_loss',
        patience=10,
        verbose=1,
        restore_best_weights=True
    )
    checkpointer = ModelCheckpoint(
        model_checkpoint_path,
        monitor='val_loss',
        verbose=1,
        save_best_only=True
    )
    return [earlystopper, checkpointer]


def train_model(model, train_gen, val_gen, X_train, X_val, batch_size=16, epochs=100, model_name="model"):
    """
    @brief Trains a Keras model using the provided generators.

    This function fits the model on the training data using data generators for augmentation,
    validates on the validation data, and applies callbacks for early stopping and model checkpointing.

    @param model (Model): Compiled Keras model to train.
    @param train_gen (generator): Training data generator.
    @param val_gen (generator): Validation data generator.
    @param X_train (numpy.ndarray): Number of training samples.
    @param X_val (numpy.ndarray): Number of validation samples.
    @param batch_size (int, optional): Number of samples per batch. Default is 16.
    @param epochs (int, optional): Maximum number of training epochs. Default is 100.
    @param model_name (str, optional): Name of the model for saving the best weights. Default is "model".

    @return History: Keras History object containing training history.
    """
    model_checkpoint_path = f'models/{model_name}_best_model.keras'
    callbacks = get_callbacks(model_checkpoint_path)

    history = model.fit(
        train_gen,
        steps_per_epoch=len(X_train) // batch_size,
        epochs=epochs,
        validation_data=val_gen,
        validation_steps=len(X_val) // batch_size,
        callbacks=callbacks
    )
    return history


def train_unet(train_gen, val_gen, X_train, X_val):
    """
    @brief Trains the U-Net model for line segmentation.

    @param train_gen (generator): Training data generator.
    @param val_gen (generator): Validation data generator.
    @param X_train (numpy.ndarray): Training data.
    @param X_val (numpy.ndarray): Validation data.

    @return History: Training history for U-Net model.
    """
    model = unet_model(input_size=(160, 160, 1))
    model = compile_model(model, loss='binary_crossentropy', learning_rate=1e-4)
    return train_model(model, train_gen, val_gen, X_train, X_val, model_name="unet")


def train_advanced_model(X_train, y_train, X_val, y_val, batch_size=32, epochs=50):
    """
    @brief Trains the Advanced Line Prediction Model for line segmentation with confidence scores.

    @param X_train (numpy.ndarray): Training images.
    @param y_train (numpy.ndarray): Training labels (coordinates + confidence scores).
    @param X_val (numpy.ndarray): Validation images.
    @param y_val (numpy.ndarray): Validation labels (coordinates + confidence scores).
    @param batch_size (int, optional): Number of samples per batch. Default is 32.
    @param epochs (int, optional): Maximum number of training epochs. Default is 50.

    @return History: Training history for Advanced Line Prediction Model.
    """
    model = create_advanced_model()
    model = compile_model(model, loss=advanced_loss, learning_rate=5e-4)

    callbacks = get_callbacks(model_checkpoint_path='models/advanced_model_best_model.keras')

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        batch_size=batch_size,
        epochs=epochs,
        callbacks=callbacks
    )
    return history


def train_regression_model(train_images, train_labels, val_images, val_labels, batch_size=32, epochs=13):
    """
    @brief Trains the regression model for numerical prediction.

    @param train_images (numpy.ndarray): Training images.
    @param train_labels (numpy.ndarray): Labels for training images.
    @param val_images (numpy.ndarray): Validation images.
    @param val_labels (numpy.ndarray): Labels for validation images.
    @param batch_size (int, optional): Number of samples per batch. Default is 32.
    @param epochs (int, optional): Maximum number of training epochs. Default is 13.

    @return History: Training history for the regression model.
    """
    model = create_regression_model(input_shape=(160, 160, 1))
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

    callbacks = get_callbacks(model_checkpoint_path='models/regression_model_best_model.keras')

    history = model.fit(
        train_images, train_labels,
        validation_data=(val_images, val_labels),
        batch_size=batch_size,
        epochs=epochs,
        callbacks=callbacks
    )
    return history


def train_enhanced_patch_transformer(X_train, y_train, X_val, y_val, batch_size=32, epochs=50):
    """
    @brief Trains the Enhanced Patch Transformer Model for line segmentation.

    @param X_train (numpy.ndarray): Training images.
    @param y_train (numpy.ndarray): Training labels (coordinates only).
    @param X_val (numpy.ndarray): Validation images.
    @param y_val (numpy.ndarray): Validation labels (coordinates only).
    @param batch_size (int, optional): Number of samples per batch. Default is 32.
    @param epochs (int, optional): Maximum number of training epochs. Default is 50.

    @return History: Training history for the Enhanced Patch Transformer Model.
    """
    model = create_enhanced_patch_transformer_model()
    model = compile_model(model, loss=scaled_loss, learning_rate=1e-3)

    callbacks = get_callbacks(model_checkpoint_path='models/enhanced_patch_transformer_best_model.keras')

    history = model.fit(
        x=X_train,
        y=y_train,
        validation_data=(X_val, y_val),
        batch_size=batch_size,
        epochs=epochs,
        callbacks=callbacks
    )
    return history
