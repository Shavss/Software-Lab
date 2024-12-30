"""
@file enhanced_patch_transformer.py
@brief Defines the Enhanced Patch Transformer model for line segmentation.

@details
This module includes:
- Custom layers (PositionalEncoding, PatchExtractor, ClipOutput)
- Transformer encoder architecture
- Enhanced Patch Transformer model for processing image patches.
- Scaled loss function for training the model.
"""

import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Conv2D, Flatten, Dense, Reshape, Layer, TimeDistributed,
    GlobalAveragePooling1D, GlobalMaxPooling1D, Add, LayerNormalization,
    Dropout, Concatenate
)
from tensorflow.keras.models import Model


# --- Custom Layers ---
class ClipOutput(Layer):
    """
    @brief Custom layer to clip model outputs to a specified range.
    
    @param clip_value_min (float): Minimum value to clip to.
    @param clip_value_max (float): Maximum value to clip to.
    """
    def __init__(self, clip_value_min=0.0, clip_value_max=1.0, **kwargs):
        super(ClipOutput, self).__init__(**kwargs)
        self.clip_value_min = clip_value_min
        self.clip_value_max = clip_value_max

    def call(self, inputs):
        return tf.clip_by_value(inputs, self.clip_value_min, self.clip_value_max)


class PositionalEncoding(Layer):
    """
    @brief Adds positional encoding to the input patches for better spatial awareness.
    
    @param num_patches (int): Number of patches in the input.
    @param d_model (int): Dimension of the embedding space for each patch.
    """
    def __init__(self, num_patches, d_model, **kwargs):
        super(PositionalEncoding, self).__init__(**kwargs)
        self.num_patches = num_patches
        self.d_model = d_model

    def call(self, inputs):
        position = tf.range(self.num_patches, dtype=tf.float32)
        div_term = tf.exp(tf.range(0., self.d_model, 2.0) * -(tf.math.log(10000.0) / self.d_model))
        pos_embedding = tf.matmul(tf.expand_dims(position, -1), div_term[tf.newaxis, :])
        pos_embedding = tf.concat([tf.sin(pos_embedding), tf.cos(pos_embedding)], axis=-1)
        return inputs + tf.expand_dims(pos_embedding, 0)


class PatchExtractor(Layer):
    """
    @brief Extracts patches from input images for patch-based processing.
    
    @param patch_size (int): Size of each patch.
    @param stride (int): Stride for extracting patches.
    """
    def __init__(self, patch_size, stride, **kwargs):
        super(PatchExtractor, self).__init__(**kwargs)
        self.patch_size = patch_size
        self.stride = stride

    def call(self, inputs):
        patches = tf.image.extract_patches(
            images=inputs,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.stride, self.stride, 1],
            rates=[1, 1, 1, 1],
            padding="VALID"
        )
        return patches


def transformer_encoder(inputs, num_heads, key_dim, ff_dim, dropout_rate=0.1):
    """
    @brief Builds a single Transformer encoder block.
    
    @param inputs (Tensor): Input tensor to the encoder block.
    @param num_heads (int): Number of attention heads.
    @param key_dim (int): Dimensionality of the attention key space.
    @param ff_dim (int): Dimensionality of the feedforward network.
    @param dropout_rate (float, optional): Dropout rate for regularization. Default is 0.1.
    
    @return Tensor: Output tensor from the Transformer encoder block.
    """
    attention_output = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=key_dim)(inputs, inputs)
    attention_output = Add()([inputs, attention_output])
    attention_output = LayerNormalization()(attention_output)

    ff_output = Dense(ff_dim, activation="relu")(attention_output)
    ff_output = Dense(inputs.shape[-1], activation="relu")(ff_output)
    ff_output = Add()([attention_output, ff_output])
    ff_output = LayerNormalization()(ff_output)

    return Dropout(dropout_rate)(ff_output)


def create_enhanced_patch_transformer_model(img_size=160, patch_size=40, stride=40):
    """
    @brief Creates the Enhanced Patch Transformer model for line segmentation.
    
    @param img_size (int, optional): Input image size (assumed square). Default is 160.
    @param patch_size (int, optional): Size of each patch. Default is 40.
    @param stride (int, optional): Stride for extracting patches. Default is 40.
    
    @return Model: Keras Model instance of the Enhanced Patch Transformer.
    """
    num_patches = (img_size // patch_size) ** 2
    image_input = Input(shape=(img_size, img_size, 1), name="input_layer")
    patches = PatchExtractor(patch_size=patch_size, stride=stride)(image_input)
    patches_reshaped = Reshape((-1, patch_size, patch_size, 1))(patches)

    # Patch Embedding via CNN Layers
    x = TimeDistributed(Conv2D(32, (3, 3), activation='relu', padding='same'))(patches_reshaped)
    x = TimeDistributed(Conv2D(64, (3, 3), activation='relu', padding='same'))(x)
    x = TimeDistributed(Flatten())(x)
    patch_features = TimeDistributed(Dense(128, activation='relu'))(x)
    patch_grid = Reshape((num_patches, 128))(patch_features)

    # Positional Encoding
    patch_grid = PositionalEncoding(num_patches, 128)(patch_grid)

    # Stacked Transformer Encoders
    for _ in range(3):
        patch_grid = transformer_encoder(patch_grid, num_heads=2, key_dim=32, ff_dim=128, dropout_rate=0.2)

    # Global Aggregation
    global_avg = GlobalAveragePooling1D()(patch_grid)
    global_max = GlobalMaxPooling1D()(patch_grid)
    global_features = Concatenate()([global_avg, global_max])

    # Output Coordinates
    line_coordinates = Dense(8 * 4, activation='linear', name="line_coordinates")(global_features)
    line_coordinates = Reshape((8, 4), name="final_output")(line_coordinates)

    # Clip output to [0, 1]
    clipped_output = ClipOutput(clip_value_min=0.0, clip_value_max=1.0)(line_coordinates)

    return Model(inputs=image_input, outputs=clipped_output)


def scaled_loss(y_true, y_pred):
    """
    @brief Computes the custom scaled loss for line prediction.
    
    @param y_true (Tensor): Ground truth line coordinates.
    @param y_pred (Tensor): Predicted line coordinates.
    
    @return Tensor: Scaled loss value.
    """
    true_coords = y_true[:, :, :4]
    pred_coords = y_pred[:, :, :4]
    coord_loss = tf.reduce_mean(tf.reduce_sum(tf.square(pred_coords - true_coords), axis=-1))
    return coord_loss
