import numpy as np
import tensorflow as tf
from tensorflow.compat.v1.layers import Conv2D
from tensorflow.compat.v1.layers import Dense
from tensorflow.nn import relu
from tensorflow.python.keras.layers import Flatten, Dropout

from models.customlayers import build_unified_encoder, build_unified_decoder


def gaussian_mixture_variational_autoencoder(x, dropout_rate, dropout, config):
    layers = {}
    # encoding network q(z|x) and q(w|x)
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

        w_mu_layer = Dense(config.dim_w)
        w_log_sigma_layer = Dense(config.dim_w)

        z_mu_layer = Dense(config.dim_z)
        z_log_sigma_layer = Dense(config.dim_z)
        dec_dense = Dense(np.prod(reshape))

        flatten = Flatten()(temp_out)

        layers['w_mu'] = w_mu = dropout_layer(w_mu_layer(flatten), dropout)
        layers['w_log_sigma'] = w_log_sigma = dropout_layer(w_log_sigma_layer(flatten), dropout)
        layers['w_sampled'] = w_sampled = w_mu + tf.random_normal(tf.shape(w_log_sigma)) * tf.exp(0.5 * w_log_sigma)

        layers['z_mu'] = z_mu = dropout_layer(z_mu_layer(flatten), dropout)
        layers['z_log_sigma'] = z_log_sigma = dropout_layer(z_log_sigma_layer(flatten))
        layers['z_sampled'] = z_sampled = z_mu + tf.random_normal(tf.shape(z_log_sigma)) * tf.exp(0.5 * z_log_sigma)

        temp_out = intermediate_conv_reverse(tf.reshape(dropout_layer(dec_dense(z_sampled), dropout), [-1, *reshape]))

    # posterior p(z|w,c)
    z_wc_mu_layer = Dense(config.dim_z * config.dim_c)
    z_wc_log_sigma_layer = Dense(config.dim_z * config.dim_c)

    z_wc_mu = z_wc_mu_layer(w_sampled)
    z_wc_log_sigma = z_wc_log_sigma_layer(w_sampled)
    z_wc_log_sigma_inv = tf.nn.bias_add(z_wc_log_sigma, bias=tf.Variable(tf.constant(0.1, shape=[z_wc_log_sigma.get_shape()[-1]], dtype=tf.float32)))
    layers['z_wc_mus'] = z_wc_mus = tf.reshape(z_wc_mu, [-1, config.dim_z, config.dim_c])
    layers['z_wc_log_sigma_invs'] = z_wc_log_sigma_invs = tf.reshape(z_wc_log_sigma_inv, [-1, config.dim_z, config.dim_c])
    # reparametrization
    layers['z_wc_sampled'] = z_wc_mus + tf.random_normal(tf.shape(z_wc_log_sigma_invs)) * tf.exp(z_wc_log_sigma_invs)

    # decoder p(x|z)
    with tf.variable_scope('Decoder'):
        decoder = build_unified_decoder(config.outputWidth, config.intermediateResolutions, config.numChannels)

        for layer in decoder:
            temp_out = layer(temp_out)

        layers['xz_mu'] = temp_out

    # prior p(c)
    z_sample = tf.tile(tf.expand_dims(z_sampled, -1), [1, 1, config.dim_c])
    loglh = -0.5 * (tf.squared_difference(z_sample, z_wc_mus) * tf.exp(z_wc_log_sigma_invs)) - z_wc_log_sigma_invs + tf.log(np.pi)
    loglh_sum = tf.reduce_sum(loglh, 1)
    layers['pc_logit'] = loglh_sum
    layers['pc'] = tf.nn.softmax(loglh_sum)

    return layers
