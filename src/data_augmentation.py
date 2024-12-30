"""
@file data_augmentation.py
@brief Module for data augmentation using Keras ImageDataGenerator.

This module provides configurations and generators for augmenting input images and masks
during the training of deep learning models. It also supports coordinate and confidence
augmentation for advanced models.
"""

from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np

def get_data_generators(X_train, y_train, X_val, y_val, batch_size=16):
    """
    @brief Creates data generators for training and validation datasets.

    This function configures ImageDataGenerators for augmenting images and masks with
    specified parameters and returns Python generators for training and validation.

    @param X_train (numpy.ndarray): Training images.
    @param y_train (numpy.ndarray): Training masks.
    @param X_val (numpy.ndarray): Validation images.
    @param y_val (numpy.ndarray): Validation masks.
    @param batch_size (int, optional): Number of samples per batch. Default is 16.

    @return tuple: Training and validation generators.
    """
    # Data Augmentation Parameters
    data_gen_args = dict(
        rotation_range=10,
        width_shift_range=0.05,
        height_shift_range=0.05,
        shear_range=0.05,
        zoom_range=0.05,
        horizontal_flip=True,
        vertical_flip=True,
        fill_mode='nearest'
    )

    image_datagen = ImageDataGenerator(**data_gen_args)
    mask_datagen = ImageDataGenerator(**data_gen_args)

    # Fit the generators
    image_datagen.fit(X_train, augment=True, seed=42)
    mask_datagen.fit(y_train, augment=True, seed=42)

    # Create generators
    def train_generator(image_datagen, mask_datagen, X, y, batch_size):
        """
        @brief Generator function for training data.

        This generator yields batches of augmented images and masks indefinitely.

        @param image_datagen (ImageDataGenerator): Data generator for images.
        @param mask_datagen (ImageDataGenerator): Data generator for masks.
        @param X (numpy.ndarray): Images data.
        @param y (numpy.ndarray): Masks data.
        @param batch_size (int): Number of samples per batch.

        @return generator: Generator yielding (images, masks) tuples.
        """
        image_generator = image_datagen.flow(X, batch_size=batch_size, seed=42)
        mask_generator = mask_datagen.flow(y, batch_size=batch_size, seed=42)
        while True:
            img = next(image_generator)
            mask = next(mask_generator)
            yield img, mask

    train_gen = train_generator(image_datagen, mask_datagen, X_train, y_train, batch_size)
    val_gen = train_generator(image_datagen, mask_datagen, X_val, y_val, batch_size)

    return train_gen, val_gen


def get_image_only_generator(X_train, X_val, batch_size=16):
    """
    @brief Creates data generators for image-only augmentation.

    This function is used for models like Enhanced Patch Transformer, which do not require
    mask augmentation but may have label transformation needs.

    @param X_train (numpy.ndarray): Training images.
    @param X_val (numpy.ndarray): Validation images.
    @param batch_size (int, optional): Number of samples per batch. Default is 16.

    @return tuple: Training and validation generators.
    """
    data_gen_args = dict(
        rotation_range=10,
        width_shift_range=0.05,
        height_shift_range=0.05,
        shear_range=0.05,
        zoom_range=0.05,
        horizontal_flip=True,
        vertical_flip=True,
        fill_mode='nearest'
    )

    image_datagen = ImageDataGenerator(**data_gen_args)

    # Fit the generator
    image_datagen.fit(X_train, augment=True, seed=42)

    # Create generators
    train_gen = image_datagen.flow(X_train, batch_size=batch_size, seed=42)
    val_gen = image_datagen.flow(X_val, batch_size=batch_size, seed=42)

    return train_gen, val_gen


def apply_transformations_to_labels(images, labels, batch_size, transformation_fn):
    """
    @brief Augments labels to match image transformations.

    This function applies user-defined transformations to labels, such as coordinates
    or confidence scores, during image augmentation.

    @param images (numpy.ndarray): Batch of images.
    @param labels (numpy.ndarray): Corresponding labels for the images.
    @param batch_size (int): Batch size for the transformations.
    @param transformation_fn (function): Transformation function for labels.

    @return tuple: Transformed images and labels.
    """
    augmented_images = []
    augmented_labels = []

    for i in range(0, len(images), batch_size):
        batch_images = images[i:i + batch_size]
        batch_labels = labels[i:i + batch_size]

        transformed_images = batch_images  # Images can be directly augmented by ImageDataGenerator
        transformed_labels = transformation_fn(batch_images, batch_labels)

        augmented_images.append(transformed_images)
        augmented_labels.append(transformed_labels)

    return np.concatenate(augmented_images), np.concatenate(augmented_labels)
