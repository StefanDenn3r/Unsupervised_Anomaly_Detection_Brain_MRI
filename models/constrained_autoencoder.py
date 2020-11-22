import numpy as np
import tensorflow as tf
from tensorflow.compat.v1.layers import Dense
from tensorflow.python.keras.layers import Conv2D, Flatten, Dropout

from models.customlayers import build_unified_encoder, build_unified_decoder


def constrained_autoencoder(x, dropout_rate, dropout, config):
    outputs = {}

    with tf.variable_scope('Encoder'):
        encoder = build_unified_encoder(x.get_shape().as_list(), config.intermediateResolutions)

        temp_out = x
        for layer in encoder:
            temp_out = layer(temp_out)

    with tf.variable_scope("Bottleneck"):
        intermediate_conv = Conv2D(temp_out.get_shape().as_list()[3] // 8, 1, padding='same')
        intermediate_conv_reverse = Conv2D(temp_out.get_shape().as_list()[3], 1, padding='same')
        dropout_layer = Dropout(dropout_rate)
        temp_out = intermediate_conv(temp_out)

        reshape = temp_out.get_shape().as_list()[1:]
        z_layer = Dense(config.zDim)
        dec_dense = Dense(np.prod(reshape))

        outputs['z'] = z = dropout_layer(z_layer(Flatten()(temp_out)), dropout)
        reshaped = tf.reshape(dropout_layer(dec_dense(z), dropout), [-1, *reshape])
        temp_out = intermediate_conv_reverse(reshaped)

    with tf.variable_scope('Decoder'):
        decoder = build_unified_decoder(config.outputWidth, config.intermediateResolutions, config.numChannels)

        # Decode: z -> x_hat
        for layer in decoder:
            temp_out = layer(temp_out)

        outputs['x_hat'] = temp_out

    with tf.variable_scope('Encoder'):
        # mapping reconstruction to latent space for constrained part
        for layer in encoder:
            temp_out = layer(temp_out)
        outputs['z_rec'] = dropout_layer(z_layer(Flatten()(intermediate_conv(temp_out))), dropout)

    return outputs
