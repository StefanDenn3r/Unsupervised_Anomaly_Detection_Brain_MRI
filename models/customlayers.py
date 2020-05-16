import math

import tensorflow as tf
from tensorflow.compat.v1.layers import Conv2D, Conv2DTranspose, BatchNormalization
from tensorflow.keras.layers import LeakyReLU, ReLU, LayerNormalization


def sample(dec_dense, decoder, reshape, tensor, zDim):
    sampled = tf.random.normal(shape=(tf.shape(tensor)[0], zDim))
    sample_out = tf.reshape(dec_dense(sampled), [-1, *reshape])
    for layer in decoder:
        sample_out = layer(sample_out)
    return sample_out


def build_unified_encoder(input_shape, intermediateResolutions, use_batchnorm=True):
    encoder = []
    num_pooling = int(math.log(input_shape[1], 2) - math.log(float(intermediateResolutions[0]), 2))
    for i in range(num_pooling):
        filters = int(min(128, 32 * (2 ** i)))
        encoder.append(Conv2D(filters=filters, kernel_size=5, strides=2, padding='same', name=f'enc_conv2D_{i}'))
        encoder.append(BatchNormalization() if use_batchnorm else LayerNormalization([1, 2]))
        encoder.append(LeakyReLU())
    return encoder


def build_unified_decoder(outputWidth, intermediateResolutions, outputChannels, final_activation=tf.identity, use_batchnorm=True):
    decoder = []
    num_upsampling = int(math.log(outputWidth, 2) - math.log(float(intermediateResolutions[0]), 2))
    decoder.append(BatchNormalization() if use_batchnorm else LayerNormalization([1, 2]))
    decoder.append(ReLU())
    for i in range(num_upsampling):
        filters = int(max(32, 128 / (2 ** i)))
        decoder.append(Conv2DTranspose(filters=filters, kernel_size=5, strides=2, padding='same', name=f'dec_Conv2DT_{i}'))
        decoder.append(BatchNormalization() if use_batchnorm else LayerNormalization([1, 2]))
        decoder.append(LeakyReLU())
    decoder.append(Conv2D(filters=outputChannels, kernel_size=1, strides=1, padding='same', name='dec_Conv2D_final', activation=final_activation))
    return decoder
