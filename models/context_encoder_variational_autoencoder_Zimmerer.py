import numpy as np
import tensorflow as tf
from tensorflow.compat.v1.layers import Conv2D, Conv2DTranspose, Flatten, Dense
from tensorflow.nn import leaky_relu


def context_encoder_variational_autoencoder_Zimmerer(x, x_ce, dropout_rate, dropout, config):
    outputs = {}
    with tf.variable_scope("Encoder"):
        enc_conv2D_1 = Conv2D(filters=16, kernel_size=4, strides=2, padding='same', name='enc_conv2D_1', activation=leaky_relu)
        enc_conv2D_2 = Conv2D(filters=64, kernel_size=4, strides=2, padding='same', name='enc_conv2D_2', activation=leaky_relu)
        enc_conv2D_3 = Conv2D(filters=256, kernel_size=4, strides=2, padding='same', name='enc_conv2D_3', activation=leaky_relu)
        enc_conv2D_4 = Conv2D(filters=1024, kernel_size=4, strides=2, padding='same', name='enc_conv2D_4', activation=leaky_relu)

        temp_out = enc_conv2D_4(enc_conv2D_3(enc_conv2D_2(enc_conv2D_1(x))))
        temp_out_ce = enc_conv2D_4(enc_conv2D_3(enc_conv2D_2(enc_conv2D_1(x_ce))))

    with tf.variable_scope("Bottleneck"):
        flatten_layer = Flatten()
        mu_layer = Dense(config.zDim)
        sigma_layer = Dense(config.zDim)
        reshape = temp_out.get_shape().as_list()[1:]

        dec_dense = Dense(np.prod(reshape))
        flatten = flatten_layer(temp_out)
        flatten_ce = flatten_layer(temp_out_ce)
        outputs['z_mu'] = z_mu = mu_layer(flatten)
        outputs['z_log_sigma'] = z_log_sigma = sigma_layer(flatten)
        outputs['z_sigma'] = z_sigma = tf.exp(z_log_sigma)
        z_vae = z_mu + tf.random_normal(tf.shape(z_sigma)) * z_sigma
        temp_out = tf.reshape(dec_dense(z_vae), [-1, *reshape])
        temp_out_ce = tf.reshape(dec_dense(mu_layer(flatten_ce)), [-1, *reshape])

    with tf.variable_scope("Decoder"):
        dec_Conv2DT_1 = Conv2DTranspose(filters=1024, kernel_size=4, strides=2, padding='same', name='dec_Conv2DT_1', activation=leaky_relu)
        dec_Conv2DT_2 = Conv2DTranspose(filters=256, kernel_size=4, strides=2, padding='same', name='dec_Conv2DT_2', activation=leaky_relu)
        dec_Conv2DT_3 = Conv2DTranspose(filters=64, kernel_size=4, strides=2, padding='same', name='dec_Conv2DT_3', activation=leaky_relu)
        dec_Conv2DT_4 = Conv2DTranspose(filters=16, kernel_size=4, strides=2, padding='same', name='dec_Conv2DT_4', activation=leaky_relu)
        dec_Conv2D_final = Conv2D(filters=1, kernel_size=4, strides=1, padding='same', name='dec_Conv2D_final')

        outputs['x_hat'] = dec_Conv2D_final(dec_Conv2DT_4(dec_Conv2DT_3(dec_Conv2DT_2(dec_Conv2DT_1(temp_out)))))

        outputs['x_hat_ce'] = dec_Conv2D_final(dec_Conv2DT_4(dec_Conv2DT_3(dec_Conv2DT_2(dec_Conv2DT_1(temp_out_ce)))))

    return outputs
