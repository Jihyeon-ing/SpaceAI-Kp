import tensorflow as tf
from tensorflow.keras import layers, Model

def tsmixer(input_length, input_channels, output_length, hidden_dim, n_blocks, dropout_rate):
    inputs = layers.Input(shape=(input_length, input_channels))
    x = inputs

    for _ in range(n_blocks):
        # Time-mixing
        y = tf.transpose(x, perm=[0, 2, 1])
        y = layers.LayerNormalization(axis=-1)(y)
        y = layers.Dense(input_length, activation='gelu')(y)
        y = layers.Dropout(dropout_rate)(y)
        y = layers.Dense(input_length, activation='gelu')(y)
        y = layers.Dropout(dropout_rate)(y)
        y = tf.transpose(y, perm=[0, 2, 1])
        x = layers.Add()([x, y])

        # Feature-mixing
        z = layers.LayerNormalization(axis=-1)(x)
        z = layers.Dense(hidden_dim, activation='gelu')(z)
        z = layers.Dropout(dropout_rate)(z)
        z = layers.Dense(input_channels, activation='gelu')(z)
        z= layers.Dropout(dropout_rate)(z)
        x = layers.Add()([x, z])

    x = tf.transpose(x, perm=[0, 2, 1])
    x = layers.Dense(output_length)(x) # time_project
    x = tf.transpose(x, perm=[0, 2, 1])
    x = layers.Dense(1)(x)
    outputs = tf.squeeze(x, axis=-1)
    return Model(inputs=inputs, outputs=outputs)
