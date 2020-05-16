import numpy as np
import tensorflow as tf
from bunch import Bunch
from tensorflow.compat.v1.layers import Conv2D, Conv2DTranspose, Dense
from tensorflow.keras.layers import ReLU, Add, LayerNormalization, AvgPool2D
from tensorflow.python.keras.layers import Flatten

from models.customlayers import build_unified_encoder


def fanogan_schlegl(z, x, dropout_rate, dropout, config):
    outputs = {}
    dim = 64
    # Encoder
    with tf.variable_scope('Encoder'):
        encoder = build_unified_encoder(x.get_shape().as_list(), intermediateResolutions=config.intermediateResolutions)
        enc_dense = Dense(config.zDim)

        temp_out = x
        for layer in encoder:
            temp_out = layer(temp_out)
        outputs['z_enc'] = z_enc = tf.nn.tanh(enc_dense(Flatten()(temp_out)))  # restricting encoder outputs to range [-1;1]

    # Generator
    with tf.variable_scope('Generator'):
        generator = Bunch({
            # Model definition
            'gen_1': Dense(np.prod(config.intermediateResolutions) * 8 * dim),

            'gen_res1_conv1': Conv2D(filters=8 * dim, kernel_size=3, padding='same'),
            'gen_res1_layernorm1': LayerNormalization([1, 2]),
            'gen_res1_conv2': Conv2DTranspose(filters=8 * dim, kernel_size=3, padding='same'),
            'gen_res1_layernorm2': LayerNormalization([1, 2]),

            'gen_res2_conv1': Conv2D(filters=4 * dim, kernel_size=3, padding='same'),
            'gen_res2_layernorm1': LayerNormalization([1, 2]),
            'gen_res2_conv2': Conv2DTranspose(filters=4 * dim, kernel_size=3, strides=2, padding='same'),
            'gen_res2_layernorm2': LayerNormalization([1, 2]),
            'gen_res2_shortcut': Conv2DTranspose(filters=4 * dim, kernel_size=1, padding='same', strides=2),

            'gen_res3_conv1': Conv2D(filters=2 * dim, kernel_size=3, padding='same'),
            'gen_res3_layernorm1': LayerNormalization([1, 2]),
            'gen_res3_conv2': Conv2DTranspose(filters=2 * dim, kernel_size=3, strides=2, padding='same'),
            'gen_res3_layernorm2': LayerNormalization([1, 2]),
            'gen_res3_shortcut': Conv2DTranspose(filters=2 * dim, kernel_size=1, padding='same', strides=2),

            'gen_res4_conv1': Conv2D(filters=1 * dim, kernel_size=3, padding='same'),
            'gen_res4_layernorm1': LayerNormalization([1, 2]),
            'gen_res4_conv2': Conv2DTranspose(filters=1 * dim, kernel_size=3, strides=2, padding='same'),
            'gen_res4_layernorm2': LayerNormalization([1, 2]),
            'gen_res4_shortcut': Conv2DTranspose(filters=1 * dim, kernel_size=1, padding='same', strides=2),

            # post process
            'gen_layernorm': LayerNormalization([1, 2]),
            'gen_conv': Conv2D(1, 1, padding='same', activation='tanh')
        })

        outputs['x_'] = x_ = evaluate_generator(generator, z, config.intermediateResolutions, dim)

        # encoder training:
        outputs['x_enc'] = x_enc = evaluate_generator(generator, z_enc, config.intermediateResolutions, dim)

    # Discriminator
    with tf.variable_scope('Discriminator'):
        discriminator = Bunch({
            # Model definition
            'dis_conv': Conv2D(dim, 3, padding='same'),

            'dis_res1_conv1': Conv2D(filters=2 * dim, kernel_size=3, padding='same'),
            'dis_res1_layernorm1': LayerNormalization([1, 2]),
            'dis_res1_conv2': Conv2D(filters=2 * dim, kernel_size=3, strides=2, padding='same'),
            'dis_res1_layernorm2': LayerNormalization([1, 2]),
            'dis_res1_shortcut1': Conv2D(filters=2 * dim, kernel_size=1, padding='same'),
            'dis_res1_shortcut2': AvgPool2D(),

            'dis_res2_conv1': Conv2D(filters=4 * dim, kernel_size=3, padding='same'),
            'dis_res2_layernorm1': LayerNormalization([1, 2]),
            'dis_res2_conv2': Conv2D(filters=4 * dim, kernel_size=3, strides=2, padding='same'),
            'dis_res2_layernorm2': LayerNormalization([1, 2]),
            'dis_res2_shortcut1': Conv2D(filters=4 * dim, kernel_size=1, padding='same'),
            'dis_res2_shortcut2': AvgPool2D(),

            'dis_res3_conv1': Conv2D(filters=8 * dim, kernel_size=3, padding='same'),
            'dis_res3_layernorm1': LayerNormalization([1, 2]),
            'dis_res3_conv2': Conv2D(filters=8 * dim, kernel_size=3, strides=2, padding='same'),
            'dis_res3_layernorm2': LayerNormalization([1, 2]),
            'dis_res3_shortcut1': Conv2D(filters=8 * dim, kernel_size=1, padding='same'),
            'dis_res3_shortcut2': AvgPool2D(),

            'dis_res4_conv1': Conv2D(filters=8 * dim, kernel_size=3, padding='same'),
            'dis_res4_layernorm1': LayerNormalization([1, 2]),
            'dis_res4_conv2': Conv2D(filters=8 * dim, kernel_size=3, padding='same'),
            'dis_res4_layernorm2': LayerNormalization([1, 2]),

            # post process
            # 'dis_flatten': Flatten(),
            'dis_dense': Dense(1),
        })

        # fake:
        outputs['d_fake_features'], outputs['d_'] = evaluate_discriminator(discriminator, x_)

        # real:
        outputs['d_features'], outputs['d'] = evaluate_discriminator(discriminator, x)

        # add noise
        alpha = tf.random_uniform(shape=[config.batchsize, 1], minval=0., maxval=1.)  # eps
        diff = tf.reshape((x_ - x), [config.batchsize, np.prod(x.get_shape().as_list()[1:])])
        outputs['x_hat'] = x_hat = x + tf.reshape(alpha * diff, [config.batchsize, *x.get_shape().as_list()[1:]])

        outputs['d_hat_features'], outputs['d_hat'] = evaluate_discriminator(discriminator, x_hat)

        # encoder training:
        outputs['d_enc_features'], outputs['d_enc'] = evaluate_discriminator(discriminator, x_enc)

    return outputs


def evaluate_generator(generator, z, intermediateResolutions, dim):
    # Evaluate
    output = tf.reshape(generator.gen_1(z), [-1, intermediateResolutions[0], intermediateResolutions[1], 8 * dim])
    # residual block 1
    output_temp = generator.gen_res1_conv2(ReLU()(generator.gen_res1_layernorm2(generator.gen_res1_conv1(ReLU()(generator.gen_res1_layernorm1(output))))))
    output = Add()([output_temp, output])
    # residual block 2
    output_temp = generator.gen_res2_conv2(ReLU()(generator.gen_res2_layernorm2(generator.gen_res2_conv1(ReLU()(generator.gen_res2_layernorm1(output))))))
    output = Add()([output_temp, generator.gen_res2_shortcut(output)])
    # residual block 3
    output_temp = generator.gen_res3_conv2(ReLU()(generator.gen_res3_layernorm2(generator.gen_res3_conv1(ReLU()(generator.gen_res3_layernorm1(output))))))
    output = Add()([output_temp, generator.gen_res3_shortcut(output)])
    # residual block 4
    output_temp = generator.gen_res4_conv2(ReLU()(generator.gen_res4_layernorm2(generator.gen_res4_conv1(ReLU()(generator.gen_res4_layernorm1(output))))))
    output = Add()([output_temp, generator.gen_res4_shortcut(output)])

    output = generator.gen_layernorm(output)
    output = ReLU()(output)
    return generator.gen_conv(output)


def evaluate_discriminator(discriminator, x):
    # Evaluate
    output = discriminator.dis_conv(x)
    # residual block 1
    output_temp = discriminator.dis_res1_conv2(
        ReLU()(discriminator.dis_res1_layernorm2(discriminator.dis_res1_conv1(ReLU()(discriminator.dis_res1_layernorm1(output))))))
    output = Add()([output_temp, discriminator.dis_res1_shortcut2(discriminator.dis_res1_shortcut1(output))])
    # residual block 2
    output_temp = discriminator.dis_res2_conv2(
        ReLU()(discriminator.dis_res2_layernorm2(discriminator.dis_res2_conv1(ReLU()(discriminator.dis_res2_layernorm1(output))))))
    output = Add()([output_temp, discriminator.dis_res2_shortcut2(discriminator.dis_res2_shortcut1(output))])
    # residual block 3
    output_temp = discriminator.dis_res3_conv2(
        ReLU()(discriminator.dis_res3_layernorm2(discriminator.dis_res3_conv1(ReLU()(discriminator.dis_res3_layernorm1(output))))))
    output = Add()([output_temp, discriminator.dis_res3_shortcut2(discriminator.dis_res3_shortcut1(output))])
    # residual block 4
    output_temp = discriminator.dis_res4_conv2(
        ReLU()(discriminator.dis_res4_layernorm2(discriminator.dis_res4_conv1(ReLU()(discriminator.dis_res4_layernorm1(output))))))
    output = Add()([output_temp, output])

    # output = discriminator.dis_flatten(output)
    return output, discriminator.dis_dense(output)
