import tensorflow as tf
from tensorflow.compat.v1.layers import Conv2D, Conv2DTranspose, Flatten, Dense
from tensorflow.nn import leaky_relu


def variational_autoencoder_Zimmerer(x, dropout_rate, dropout, config):
    outputs = {}
    enc_conv2D_1 = Conv2D(filters=16, kernel_size=4, strides=2, padding='same', name='enc_conv2D_1', activation=leaky_relu)
    enc_conv2D_2 = Conv2D(filters=64, kernel_size=4, strides=2, padding='same', name='enc_conv2D_2', activation=leaky_relu)
    enc_conv2D_3 = Conv2D(filters=256, kernel_size=4, strides=2, padding='same', name='enc_conv2D_3', activation=leaky_relu)
    enc_conv2D_4 = Conv2D(filters=1024, kernel_size=4, strides=2, padding='same', name='enc_conv2D_4', activation=leaky_relu)
    flatten_layer = Flatten()
    mu_layer = Dense(config.zDim)
    sigma_layer = Dense(config.zDim)

    dec_dense = Dense(config.intermediateResolutions[0] * config.intermediateResolutions[1] * 1024)
    dec_Conv2DT_1 = Conv2DTranspose(filters=1024, kernel_size=4, strides=2, padding='same', name='dec_Conv2DT_1', activation=leaky_relu)
    dec_Conv2DT_2 = Conv2DTranspose(filters=256, kernel_size=4, strides=2, padding='same', name='dec_Conv2DT_2', activation=leaky_relu)
    dec_Conv2DT_3 = Conv2DTranspose(filters=64, kernel_size=4, strides=2, padding='same', name='dec_Conv2DT_3', activation=leaky_relu)
    dec_Conv2DT_4 = Conv2DTranspose(filters=16, kernel_size=4, strides=2, padding='same', name='dec_Conv2DT_4', activation=leaky_relu)
    dec_Conv2D_final = Conv2D(filters=1, kernel_size=4, strides=1, padding='same', name='dec_Conv2D_final')

    flatten = flatten_layer(enc_conv2D_4(enc_conv2D_3(enc_conv2D_2(enc_conv2D_1(x)))))
    outputs['z_mu'] = z_mu = mu_layer(flatten)
    outputs['z_log_sigma'] = z_log_sigma = sigma_layer(flatten)
    outputs['z_sigma'] = z_sigma = tf.exp(z_log_sigma)
    z_vae = z_mu + tf.random_normal(tf.shape(z_sigma)) * z_sigma
    # Decode: z -> x_hat
    reshaped = tf.reshape(dec_dense(z_vae), [-1, config.intermediateResolutions[0], config.intermediateResolutions[1], 1024])
    outputs['x_hat'] = dec_Conv2D_final(dec_Conv2DT_4(dec_Conv2DT_3(dec_Conv2DT_2(dec_Conv2DT_1(reshaped)))))

    return outputs
