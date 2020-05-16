import numpy as np
import tensorflow as tf
from tensorflow import sigmoid
from tensorflow.compat.v1.layers import Conv2D, Flatten
from tensorflow.compat.v1.layers import Dense
from tensorflow.python.keras.layers import Conv2D, Dropout, Flatten

from models.customlayers import build_unified_decoder, build_unified_encoder


def fanogan(z, x, dropout_rate, dropout, config):
    outputs = {}

    # Encoder
    with tf.variable_scope('Encoder'):
        encoder = build_unified_encoder(x.get_shape().as_list(), config.intermediateResolutions)

        temp_out = x
        for layer in encoder:
            temp_out = layer(temp_out)

        temp_temp_out = temp_out
        intermediate_conv = Conv2D(temp_temp_out.get_shape().as_list()[3] // 8, 1, padding='same')
        dropout_layer = Dropout(dropout_rate)
        temp_out = intermediate_conv(temp_out)

        reshape = temp_out.get_shape().as_list()[1:]
        z_layer = Dense(config.zDim)
        outputs['z_enc'] = z_enc = tf.nn.tanh(dropout_layer(z_layer(Flatten()(temp_out)), dropout))

    # Generator
    with tf.variable_scope('Generator'):
        intermediate_conv_reverse = Conv2D(temp_temp_out.get_shape().as_list()[3], 1, padding='same')
        dec_dense = Dense(np.prod(reshape))
        generator = build_unified_decoder(config.outputWidth, config.intermediateResolutions, config.numChannels, use_batchnorm=False)

        temp_out_z_enc = intermediate_conv_reverse(tf.reshape(dropout_layer(dec_dense(z_enc), dropout), [-1, *reshape]))
        # encoder training:
        for layer in generator:
            temp_out_z_enc = layer(temp_out_z_enc)
        outputs['x_enc'] = x_enc = sigmoid(temp_out_z_enc)  # recon_img
        # generator training
        temp_out = intermediate_conv_reverse(tf.reshape(dropout_layer(dec_dense(z), dropout), [-1, *reshape]))
        for layer in generator:
            temp_out = layer(temp_out)
        outputs['x_'] = x_ = sigmoid(temp_out)

    # Discriminator
    with tf.variable_scope('Discriminator'):
        discriminator = build_unified_encoder(x_.get_shape().as_list(), config.intermediateResolutions, use_batchnorm=False)
        discriminator_dense = Dense(1)

        # fake:
        temp_out = x_
        for layer in discriminator:
            temp_out = layer(temp_out)
        outputs['d_fake_features'] = temp_out
        outputs['d_'] = discriminator_dense(temp_out)

        # real:
        temp_out = x
        for layer in discriminator:
            temp_out = layer(temp_out)
        outputs['d_features'] = temp_out  # image_features
        outputs['d'] = discriminator_dense(temp_out)

        alpha = tf.random_uniform(shape=[config.batchsize, 1], minval=0., maxval=1.)  # eps
        diff = tf.reshape((x_ - x), [config.batchsize, np.prod(x.get_shape().as_list()[1:])])
        outputs['x_hat'] = x_hat = x + tf.reshape(alpha * diff, [config.batchsize, *x.get_shape().as_list()[1:]])

        temp_out = x_hat
        for layer in discriminator:
            temp_out = layer(temp_out)
        outputs['d_hat_features'] = temp_out
        outputs['d_hat'] = discriminator_dense(temp_out)

        # encoder training:
        temp_out = x_enc
        for layer in discriminator:
            temp_out = layer(temp_out)
        outputs['d_enc_features'] = temp_out  # recon_features
        outputs['d_enc'] = discriminator_dense(temp_out)

    return outputs
