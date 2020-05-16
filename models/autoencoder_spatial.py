import tensorflow as tf
from tensorflow.python.keras.layers import Dropout

from models.customlayers import build_unified_encoder, build_unified_decoder


def autoencoder_spatial(x, dropout_rate, dropout, config):
    outputs = {}

    with tf.variable_scope('Encoder'):
        encoder = build_unified_encoder(x.get_shape().as_list(), config.intermediateResolutions)
        dropout_layer = Dropout(dropout_rate)
        temp_out = x
        for layer in encoder:
            temp_out = layer(temp_out)
        temp_out = dropout_layer(temp_out, training=dropout)
    outputs['z'] = temp_out

    with tf.variable_scope('Decoder'):
        decoder = build_unified_decoder(config.outputWidth, config.intermediateResolutions, config.numChannels)
        # Decode: z -> x_hat
        for layer in decoder:
            temp_out = layer(temp_out)

        outputs['x_hat'] = temp_out

    return outputs
