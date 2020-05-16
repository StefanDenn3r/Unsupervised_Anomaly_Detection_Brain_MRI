import numpy as np
import tensorflow as tf
from tensorflow.compat.v1.layers import Conv2D, Flatten
from tensorflow.compat.v1.layers import Dense
from tensorflow.python.keras.layers import Conv2D, Dropout, Flatten

from models.customlayers import build_unified_decoder, build_unified_encoder


def anovaegan(x, dropout_rate, dropout, config):
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
        mu_layer = Dense(config.zDim)
        sigma_layer = Dense(config.zDim)

        flatten = Flatten()(temp_out)
        outputs['z_mu'] = z_mu = dropout_layer(mu_layer(flatten), dropout)
        outputs['z_log_sigma'] = z_log_sigma = dropout_layer(sigma_layer(flatten), dropout)
        outputs['z_sigma'] = z_sigma = tf.exp(z_log_sigma)
        z_vae = z_mu + tf.random_normal(tf.shape(z_sigma)) * z_sigma

    with tf.variable_scope("Generator"):
        intermediate_conv_reverse = Conv2D(temp_temp_out.get_shape().as_list()[3], 1, padding='same')
        dec_dense = Dense(np.prod(reshape))
        decoder = build_unified_decoder(outputWidth=config.outputWidth, intermediateResolutions=config.intermediateResolutions,
                                        outputChannels=config.numChannels,
                                        use_batchnorm=False)

        reshaped = tf.reshape(dropout_layer(dec_dense(z_vae)), [-1, *reshape])
        temp_out = intermediate_conv_reverse(reshaped)

        # Decode: z -> x_hat
        for layer in decoder:
            temp_out = layer(temp_out)

        outputs['out'] = temp_out

    # Discriminator
    with tf.variable_scope('Discriminator'):
        discriminator = build_unified_encoder(temp_out.get_shape().as_list(), intermediateResolutions=config.intermediateResolutions, use_batchnorm=False)
        discriminator_dense = Dense(1)

        # fake/reconstructed:
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

        # for GP
        alpha = tf.random_uniform(shape=[config.batchsize, 1], minval=0., maxval=1.)  # eps
        diff = tf.reshape((outputs['out'] - x), [config.batchsize, np.prod(x.get_shape().as_list()[1:])])
        outputs['x_hat'] = x_hat = x + tf.reshape(alpha * diff, [config.batchsize, *x.get_shape().as_list()[1:]])

        temp_out = x_hat
        for layer in discriminator:
            temp_out = layer(temp_out)
        outputs['d_hat_features'] = temp_out
        outputs['d_hat'] = discriminator_dense(temp_out)
    return outputs
