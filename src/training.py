"""
@file training.py
@brief Module for compiling and training the U-Net model.

This module provides functions to compile the U-Net model, set up callbacks,
and train the model using the provided data generators.
"""

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

def compile_model(model, learning_rate=1e-4):
    """
    @brief Compiles the U-Net model with specified optimizer and loss function.

    This function configures the model for training by setting the optimizer, loss,
    and evaluation metrics.

    @param model (Model): Keras model to compile.
    @param learning_rate (float, optional): Learning rate for the optimizer. Default is 1e-4.

    @return Model: Compiled Keras model.
    """
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model

def get_callbacks(model_checkpoint_path='models/unet_best_model.keras'):
    """
    @brief Creates callbacks for training.

    This function sets up EarlyStopping to halt training when validation loss stops improving
    and ModelCheckpoint to save the best model during training.

    @param model_checkpoint_path (str, optional): Path to save the best model. Default is 'unet_best_model.keras'.

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

def train_model(model, train_gen, val_gen, X_train, X_val, batch_size=16, epochs=100):
    """
    @brief Trains the U-Net model using the provided generators.

    This function fits the model on the training data using data generators for augmentation,
    validates on the validation data, and applies callbacks for early stopping and model checkpointing.

    @param model (Model): Compiled Keras model to train.
    @param train_gen (generator): Training data generator.
    @param val_gen (generator): Validation data generator.
    @param X_train (numpy.ndarray): Number of training samples.
    @param X_val (numpy.ndarray): Number of validation samples.
    @param batch_size (int, optional): Number of samples per batch. Default is 16.
    @param epochs (int, optional): Maximum number of training epochs. Default is 100.

    @return History: Keras History object containing training history.
    """
    callbacks = get_callbacks()

    history = model.fit(
        train_gen,
        steps_per_epoch=len(X_train) // batch_size,
        epochs=epochs,
        validation_data=val_gen,
        validation_steps=len(X_val) // batch_size,
        callbacks=callbacks
    )
    return history
