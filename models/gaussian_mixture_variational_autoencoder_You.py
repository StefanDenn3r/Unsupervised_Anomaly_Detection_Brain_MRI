import numpy as np
import tensorflow as tf
from tensorflow.compat.v1.layers import Conv2D, Conv2DTranspose
from tensorflow.image import ResizeMethod
from tensorflow.nn import relu


def gaussian_mixture_variational_autoencoder_You(x, dropout_rate, dropout, config):
    outputs = {}

    # encoding network q(z|x) and q(w|x)
    conv_1 = Conv2D(filters=64, kernel_size=3, strides=2, padding='same', name='q_wz_x/3x3convlayer', activation=relu)  # valid
    conv_2 = Conv2D(filters=64, kernel_size=3, strides=1, padding='same', name='q_wz_x/3x3convlayer1', activation=relu)  # valid
    conv_3 = Conv2D(filters=64, kernel_size=3, strides=1, padding='same', name='q_wz_x/3x3convlayer2', activation=relu)  # valid
    conv_4 = Conv2D(filters=64, kernel_size=3, strides=2, padding='same', name='q_wz_x/3x3convlayer3', activation=relu)  # valid
    conv_5 = Conv2D(filters=64, kernel_size=3, strides=1, padding='same', name='q_wz_x/3x3convlayer4', activation=relu)  # valid
    conv_6 = Conv2D(filters=64, kernel_size=3, strides=1, padding='same', name='q_wz_x/3x3convlayer5', activation=relu)  # valid

    w_mu_layer = Conv2D(filters=config.dim_w, kernel_size=1, strides=1, padding='same', name='q_wz_x/w_mu')  # valid
    w_log_sigma_layer = Conv2D(filters=config.dim_w, kernel_size=1, strides=1, padding='same', name='q_wz_x/w_log_sigma')  # valid

    z_mu_layer = Conv2D(filters=config.dim_z, kernel_size=1, strides=1, padding='same', name='q_wz_x/z_mu')  # valid
    z_log_sigma_layer = Conv2D(filters=config.dim_z, kernel_size=1, strides=1, padding='same', name='q_wz_x/z_log_sigma')  # valid

    enc = conv_6(conv_5(conv_4(conv_3(conv_2(conv_1(x))))))
    outputs['w_mu'] = w_mu = w_mu_layer(enc)
    outputs['w_log_sigma'] = w_log_sigma = w_log_sigma_layer(enc)
    # reparametrization
    outputs['w_sampled'] = w_sampled = w_mu + tf.random_normal(tf.shape(w_log_sigma)) * tf.exp(0.5 * w_log_sigma)

    outputs['z_mu'] = z_mu = z_mu_layer(enc)
    outputs['z_log_sigma'] = z_log_sigma = z_log_sigma_layer(enc)
    # reparametrization
    outputs['z_sampled'] = z_sampled = z_mu + tf.random_normal(tf.shape(z_log_sigma)) * tf.exp(0.5 * z_log_sigma)

    # posterior p(z|w,c)
    conv_7 = Conv2D(filters=64, kernel_size=1, strides=1, padding='same', name='p_z_wc/1x1convlayer', activation=relu)  # valid
    z_wc_mu_layer = Conv2D(filters=config.dim_z * config.dim_c, kernel_size=1, strides=1, padding='same', name='p_z_wc/z_wc_mu')  # valid
    z_wc_log_sigma_layer = Conv2D(filters=config.dim_z * config.dim_c, kernel_size=1, strides=1, padding='same', name='p_z_wc/z_wc_log_sigma')  # valid

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
    conv_8 = Conv2D(filters=64, kernel_size=3, strides=1, padding='same', name='p_x_z/3x3convlayer1', activation=relu)
    transpose_conv_1 = Conv2DTranspose(filters=64, kernel_size=3, strides=1, padding='same', name='p_x_z/3x3upconvlayer1', activation=relu)  # valid
    transpose_conv_2 = Conv2DTranspose(filters=64, kernel_size=3, strides=1, padding='same', name='p_x_z/3x3upconvlayer2', activation=relu)  # valid
    # resize_image, upsampling
    conv_9 = Conv2D(filters=64, kernel_size=3, strides=1, padding='same', name='p_x_z/3x3convlayer2', activation=relu)
    transpose_conv_3 = Conv2DTranspose(filters=64, kernel_size=3, strides=1, padding='same', name='p_x_z/3x3upconvlayer3', activation=relu)  # valid
    transpose_conv_4 = Conv2DTranspose(filters=64, kernel_size=3, strides=1, padding='same', name='p_x_z/3x3upconvlayer4', activation=relu)  # valid
    # resize_image, upsampling
    conv_10 = Conv2D(filters=64, kernel_size=3, strides=1, padding='same', name='p_x_z/3x3convlayer3')

    xz_mu_layer = Conv2D(filters=1, kernel_size=3, strides=1, padding='same', name='p_x_z/y_mu')  # valid

    dec_part1 = transpose_conv_2(transpose_conv_1(conv_8(z_sampled)))
    _, height1, width1, _ = dec_part1.get_shape().as_list()
    dec_part1_reshaped = tf.image.resize_images(dec_part1, (2 * height1, 2 * width1), ResizeMethod.NEAREST_NEIGHBOR)
    dec_part1_padded = dec_part1_reshaped
    dec_part2 = transpose_conv_4(transpose_conv_3(conv_9(dec_part1_padded)))
    _, height2, width2, _ = dec_part2.get_shape().as_list()
    dec_part2_reshaped = tf.image.resize_images(dec_part2, (2 * height2, 2 * width2), ResizeMethod.NEAREST_NEIGHBOR)
    dec_part2_padded = dec_part2_reshaped

    dec = conv_10(dec_part2_padded)

    outputs['xz_mu'] = xz_mu_layer(dec)

    # prior p(c)
    z_sample = tf.tile(tf.expand_dims(z_sampled, -1), [1, 1, 1, 1, config.dim_c])
    loglh = -0.5 * (tf.squared_difference(z_sample, z_wc_mus) * tf.exp(z_wc_log_sigma_invs)) - z_wc_log_sigma_invs + tf.log(np.pi)
    loglh_sum = tf.reduce_sum(loglh, 3)
    outputs['pc_logit'] = loglh_sum
    outputs['pc'] = tf.nn.softmax(loglh_sum)

    return outputs
