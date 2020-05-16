import numpy as np
import tensorflow as tf
from bunch import Bunch
from tensorflow.compat.v1.layers import Dense
from tensorflow.nn import leaky_relu
from tensorflow.python.keras.layers import AvgPool2D, ReLU, Add, LayerNormalization
from tensorflow.python.layers.convolutional import Conv2D, Conv2DTranspose
from tensorflow.python.layers.core import Flatten


def constrained_adversarial_autoencoder_Chen(z, x, dropout_rate, dropout, config):
    outputs = {}
    dim = 64
    with tf.variable_scope('Encoder'):
        encoder = Bunch({
            # Model definition
            'enc_conv': Conv2D(filters=dim, kernel_size=3, padding='same'),

            'enc_res1_conv1': Conv2D(filters=2 * dim, kernel_size=3, padding='same'),
            'enc_res1_layernorm1': LayerNormalization([1, 2]),
            'enc_res1_conv2': Conv2D(filters=2 * dim, kernel_size=3, strides=2, padding='same'),
            'enc_res1_layernorm2': LayerNormalization([1, 2]),
            'enc_res1_shortcut1': Conv2D(filters=2 * dim, kernel_size=1, padding='same'),
            'enc_res1_shortcut2': AvgPool2D(),

            'enc_res2_conv1': Conv2D(filters=4 * dim, kernel_size=3, padding='same'),
            'enc_res2_layernorm1': LayerNormalization([1, 2]),
            'enc_res2_conv2': Conv2D(filters=4 * dim, kernel_size=3, strides=2, padding='same'),
            'enc_res2_layernorm2': LayerNormalization([1, 2]),
            'enc_res2_shortcut1': Conv2D(filters=4 * dim, kernel_size=1, padding='same'),
            'enc_res2_shortcut2': AvgPool2D(),

            'enc_res3_conv1': Conv2D(filters=8 * dim, kernel_size=3, padding='same'),
            'enc_res3_layernorm1': LayerNormalization([1, 2]),
            'enc_res3_conv2': Conv2D(filters=8 * dim, kernel_size=3, strides=2, padding='same'),
            'enc_res3_layernorm2': LayerNormalization([1, 2]),
            'enc_res3_shortcut1': Conv2D(filters=8 * dim, kernel_size=1, padding='same'),
            'enc_res3_shortcut2': AvgPool2D(),

            'enc_res4_conv1': Conv2D(filters=8 * dim, kernel_size=3, padding='same'),
            'enc_res4_layernorm1': LayerNormalization([1, 2]),
            'enc_res4_conv2': Conv2D(filters=8 * dim, kernel_size=3, padding='same'),
            'enc_res4_layernorm2': LayerNormalization([1, 2]),

            'enc_flatten': Flatten(),
            'enc_dense': Dense(config.zDim),
        })
        features, z_ = evaluate_encoder(encoder, x)
        outputs['z_'] = z_

    with tf.variable_scope('Decoder'):
        decoder = Bunch({
            # Model definition
            'dec_1': Dense(np.prod(features.get_shape().as_list()[1:])),

            'dec_res1_conv1': Conv2D(filters=8 * dim, kernel_size=3, padding='same'),
            'dec_res1_layernorm1': LayerNormalization([1, 2]),
            'dec_res1_conv2': Conv2DTranspose(filters=8 * dim, kernel_size=3, padding='same'),
            'dec_res1_layernorm2': LayerNormalization([1, 2]),

            'dec_res2_conv1': Conv2D(filters=4 * dim, kernel_size=3, padding='same'),
            'dec_res2_layernorm1': LayerNormalization([1, 2]),
            'dec_res2_conv2': Conv2DTranspose(filters=4 * dim, kernel_size=3, strides=2, padding='same'),
            'dec_res2_layernorm2': LayerNormalization([1, 2]),
            'dec_res2_shortcut': Conv2DTranspose(filters=4 * dim, kernel_size=1, padding='same', strides=2),

            'dec_res3_conv1': Conv2D(filters=2 * dim, kernel_size=3, padding='same'),
            'dec_res3_layernorm1': LayerNormalization([1, 2]),
            'dec_res3_conv2': Conv2DTranspose(filters=2 * dim, kernel_size=3, strides=2, padding='same'),
            'dec_res3_layernorm2': LayerNormalization([1, 2]),
            'dec_res3_shortcut': Conv2DTranspose(filters=2 * dim, kernel_size=1, padding='same', strides=2),

            'dec_res4_conv1': Conv2D(filters=dim, kernel_size=3, padding='same'),
            'dec_res4_layernorm1': LayerNormalization([1, 2]),
            'dec_res4_conv2': Conv2DTranspose(filters=dim, kernel_size=3, strides=2, padding='same'),
            'dec_res4_layernorm2': LayerNormalization([1, 2]),
            'dec_res4_shortcut': Conv2DTranspose(filters=dim, kernel_size=1, padding='same', strides=2),

            # post process
            'dec_layernorm': LayerNormalization([1, 2]),
            'dec_conv': Conv2D(1, 1, padding='same'),
        })
        outputs['x_hat'] = x_hat = evaluate_decoder(decoder, z_, features.get_shape().as_list()[1:])

    # projecting reconstruction to latent space for constrained part
    outputs['z_rec'] = evaluate_encoder(encoder, x_hat)[1]

    # Discriminator
    with tf.variable_scope('Discriminator'):
        discriminator = [
            Dense(400, activation=leaky_relu),
            Dense(200, activation=leaky_relu),
            Dense(1)
        ]

        # fake
        temp_out = z_
        for layer in discriminator:
            temp_out = layer(temp_out)
        outputs['d_'] = temp_out

        # real
        temp_out = z
        for layer in discriminator:
            temp_out = layer(temp_out)
        outputs['d'] = temp_out

        # reparametrization
        epsilon = tf.random_uniform([], 0.0, 1.0)
        outputs['z_hat'] = z_hat = epsilon * z + (1 - epsilon) * z_

        temp_out = z_hat
        for layer in discriminator:
            temp_out = layer(temp_out)
        outputs['d_hat'] = temp_out

    return outputs


def evaluate_encoder(encoder, x):
    # Evaluate
    output = encoder.enc_conv(x)
    # residual block 1
    output_temp = encoder.enc_res1_conv2(
        ReLU()(encoder.enc_res1_layernorm2(encoder.enc_res1_conv1(ReLU()(encoder.enc_res1_layernorm1(output))))))
    output = Add()([output_temp, encoder.enc_res1_shortcut2(encoder.enc_res1_shortcut1(output))])
    # residual block 2
    output_temp = encoder.enc_res2_conv2(
        ReLU()(encoder.enc_res2_layernorm2(encoder.enc_res2_conv1(ReLU()(encoder.enc_res2_layernorm1(output))))))
    output = Add()([output_temp, encoder.enc_res2_shortcut2(encoder.enc_res2_shortcut1(output))])
    # residual block 3
    output_temp = encoder.enc_res3_conv2(
        ReLU()(encoder.enc_res3_layernorm2(encoder.enc_res3_conv1(ReLU()(encoder.enc_res3_layernorm1(output))))))
    output = Add()([output_temp, encoder.enc_res3_shortcut2(encoder.enc_res3_shortcut1(output))])
    # residual block 4
    output_temp = encoder.enc_res4_conv2(
        ReLU()(encoder.enc_res4_layernorm2(encoder.enc_res4_conv1(ReLU()(encoder.enc_res4_layernorm1(output))))))
    output = Add()([output_temp, output])

    flatten = encoder.enc_flatten(output)
    return output, encoder.enc_dense(flatten)


def evaluate_decoder(decoder, z, reshape):
    # Evaluate
    output = tf.reshape(decoder.dec_1(z), [-1, *reshape])
    # residual block 1
    output_temp = decoder.dec_res1_conv2(ReLU()(decoder.dec_res1_layernorm2(decoder.dec_res1_conv1(ReLU()(decoder.dec_res1_layernorm1(output))))))
    output = Add()([output_temp, output])
    # residual block 2
    output_temp = decoder.dec_res2_conv2(ReLU()(decoder.dec_res2_layernorm2(decoder.dec_res2_conv1(ReLU()(decoder.dec_res2_layernorm1(output))))))
    output = Add()([output_temp, decoder.dec_res2_shortcut(output)])
    # residual block 3
    output_temp = decoder.dec_res3_conv2(ReLU()(decoder.dec_res3_layernorm2(decoder.dec_res3_conv1(ReLU()(decoder.dec_res3_layernorm1(output))))))
    output = Add()([output_temp, decoder.dec_res3_shortcut(output)])
    # residual block 4
    output_temp = decoder.dec_res4_conv2(ReLU()(decoder.dec_res4_layernorm2(decoder.dec_res4_conv1(ReLU()(decoder.dec_res4_layernorm1(output))))))
    output = Add()([output_temp, decoder.dec_res4_shortcut(output)])

    output = decoder.dec_layernorm(output)
    output = ReLU()(output)
    return decoder.dec_conv(output)
