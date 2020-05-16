import numpy as np
import tensorflow as tf
from tensorflow.compat.v1.layers import Conv2D
from tensorflow.nn import relu

from models.customlayers import build_unified_encoder, build_unified_decoder


def gaussian_mixture_variational_autoencoder_spatial(x, dropout_rate, dropout, config):
    outputs = {}

    # encoding network q(z|x) and q(w|x)
    encoder = build_unified_encoder(x.get_shape().as_list(), config.intermediateResolutions)

    w_mu_layer = Conv2D(filters=config.dim_w, kernel_size=1, strides=1, padding='same', name='q_wz_x/w_mu')
    w_log_sigma_layer = Conv2D(filters=config.dim_w, kernel_size=1, strides=1, padding='same', name='q_wz_x/w_log_sigma')

    z_mu_layer = Conv2D(filters=config.dim_z, kernel_size=1, strides=1, padding='same', name='q_wz_x/z_mu')
    z_log_sigma_layer = Conv2D(filters=config.dim_z, kernel_size=1, strides=1, padding='same', name='q_wz_x/z_log_sigma')

    temp_out = x
    for layer in encoder:
        temp_out = layer(temp_out)

    outputs['w_mu'] = w_mu = w_mu_layer(temp_out)
    outputs['w_log_sigma'] = w_log_sigma = w_log_sigma_layer(temp_out)
    # reparametrization
    outputs['w_sampled'] = w_sampled = w_mu + tf.random_normal(tf.shape(w_log_sigma)) * tf.exp(0.5 * w_log_sigma)

    outputs['z_mu'] = z_mu = z_mu_layer(temp_out)
    outputs['z_log_sigma'] = z_log_sigma = z_log_sigma_layer(temp_out)
    # reparametrization
    outputs['z_sampled'] = z_sampled = z_mu + tf.random_normal(tf.shape(z_log_sigma)) * tf.exp(0.5 * z_log_sigma)

    # posterior p(z|w,c)
    conv_7 = Conv2D(filters=64, kernel_size=1, strides=1, padding='same', name='p_z_wc/1x1convlayer', activation=relu)
    z_wc_mu_layer = Conv2D(filters=config.dim_z * config.dim_c, kernel_size=1, strides=1, padding='same', name='p_z_wc/z_wc_mu')
    z_wc_log_sigma_layer = Conv2D(filters=config.dim_z * config.dim_c, kernel_size=1, strides=1, padding='same', name='p_z_wc/z_wc_log_sigma')

    mid = conv_7(w_sampled)
    z_wc_mu = z_wc_mu_layer(mid)
    z_wc_log_sigma = z_wc_log_sigma_layer(mid)
    z_wc_log_sigma_inv = tf.nn.bias_add(z_wc_log_sigma, bias=tf.Variable(tf.constant(0.1, shape=[z_wc_log_sigma.get_shape()[-1]], dtype=tf.float32)))
    outputs['z_wc_mus'] = z_wc_mus = tf.reshape(z_wc_mu, [-1, z_wc_mu.get_shape().as_list()[1], z_wc_mu.get_shape().as_list()[2], config.dim_z, config.dim_c])
    z_wc_sigma_shape = z_wc_log_sigma_inv.get_shape().as_list()
    outputs['z_wc_log_sigma_invs'] = z_wc_log_sigma_invs = tf.reshape(z_wc_log_sigma_inv,
                                                                      [-1, z_wc_sigma_shape[1], z_wc_sigma_shape[2], config.dim_z, config.dim_c])
    # reparametrization
    outputs['z_wc_sampled'] = z_wc_mus + tf.random_normal(tf.shape(z_wc_log_sigma_invs)) * tf.exp(z_wc_log_sigma_invs)

    # decoder p(x|z)
    decoder = build_unified_decoder(config.outputWidth, config.intermediateResolutions, config.numChannels)
    for layer in decoder:
        temp_out = layer(temp_out)

    outputs['xz_mu'] = temp_out

    # prior p(c)
    z_sample = tf.tile(tf.expand_dims(z_sampled, -1), [1, 1, 1, 1, config.dim_c])
    loglh = -0.5 * (tf.squared_difference(z_sample, z_wc_mus) * tf.exp(z_wc_log_sigma_invs)) - z_wc_log_sigma_invs + tf.log(np.pi)
    loglh_sum = tf.reduce_sum(loglh, 3)
    outputs['pc_logit'] = loglh_sum
    outputs['pc'] = tf.nn.softmax(loglh_sum)

    return outputs
