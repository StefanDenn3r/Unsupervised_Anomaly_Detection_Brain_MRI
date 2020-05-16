from collections import defaultdict
from math import inf

from tensorflow.python.ops.losses.losses_impl import Reduction

from trainers import trainer_utils
from trainers.AEMODEL import AEMODEL, Phase, indicate_early_stopping, update_log_dicts
from trainers.DLMODEL import *


class AnoVAEGAN(AEMODEL):
    class Config(AEMODEL.Config):
        def __init__(self):
            super().__init__('AnoVAEGAN')
            self.scale = 10.0
            self.kappa = 1.0
            self.kl_weight = 1.0

    def __init__(self, sess, config, network=None):
        super().__init__(sess, config, network)
        self.x = tf.placeholder(tf.float32, [None, self.config.outputHeight, self.config.outputWidth, self.config.numChannels], name='x')
        self.z = tf.placeholder(tf.float32, [None, self.config.zDim], name='z')

        self.outputs = self.network(self.x, dropout_rate=self.dropout_rate, dropout=self.dropout, config=self.config)

        self.reconstruction = self.outputs['out']
        self.z_mu = self.outputs['z_mu']
        self.z_sigma = self.outputs['z_sigma']
        self.d_fake_features = self.outputs['d_fake_features']
        self.d_ = self.outputs['d_']
        self.d_features = self.outputs['d_features']
        self.d = self.outputs['d']
        self.x_hat = self.outputs['x_hat']
        self.d_hat = self.outputs['d_hat']

        self.kappa = self.config.kappa
        self.kl_weight = self.config.kl_weight
        self.scale = self.config.scale

        # Print Stats
        self.get_number_of_trainable_params()
        # Instantiate Saver
        self.saver = tf.train.Saver()

    def train(self, dataset):
        # Determine trainable variables
        self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)

        # Build losses
        self.losses['disc_fake'] = disc_fake = tf.reduce_mean(self.d_)
        self.losses['disc_real'] = disc_real = tf.reduce_mean(self.d)
        disc_loss = disc_fake - disc_real

        ddx = tf.gradients(self.d_hat, self.x_hat)[0]  # gradient
        ddx = tf.sqrt(tf.reduce_sum(tf.square(ddx), axis=1))  # slopes
        ddx = tf.reduce_mean(tf.square(ddx - 1.0)) * self.scale  # gradient penalty
        self.losses['disc_loss'] = disc_loss = disc_loss + ddx

        # Build losses
        kl = 0.5 * tf.reduce_sum(tf.square(self.z_mu) + tf.square(self.z_sigma) - tf.log(tf.square(self.z_sigma)) - 1,
                                 axis=1)
        self.losses['kl'] = loss_kl = tf.reduce_mean(kl)

        self.losses['loss_img'] = tf.reduce_mean(
            tf.reduce_mean(tf.losses.mean_squared_error(self.x, self.reconstruction, reduction=Reduction.NONE), axis=[1, 2, 3]))
        self.losses['loss_fts'] = tf.reduce_mean(
            tf.reduce_mean(tf.losses.mean_squared_error(self.d_fake_features, self.d_features, reduction=Reduction.NONE), axis=[1, 2, 3]))
        self.losses['L1'] = tf.losses.absolute_difference(self.x, self.reconstruction, reduction=Reduction.NONE)
        self.losses['reconstructionLoss'] = self.losses['loss'] = tf.reduce_mean(tf.reduce_sum(self.losses['L1'], axis=[1, 2, 3]))

        self.losses['gen_loss'] = gen_loss = - disc_fake
        self.losses['enc_loss'] = enc_loss = self.losses['reconstructionLoss'] + self.kl_weight * loss_kl

        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            # Set the optimizer
            t_vars = tf.trainable_variables()
            dis_vars = [var for var in t_vars if 'Discriminator' in var.name]
            gen_vars = [var for var in t_vars if 'Generator' in var.name]
            enc_vars = [var for var in t_vars if 'Encoder' in var.name]

            optim_dis = tf.train.AdamOptimizer(learning_rate=self.config.learningrate, beta1=0.5, beta2=0.9).minimize(disc_loss, var_list=dis_vars)
            optim_gen = tf.train.AdamOptimizer(learning_rate=self.config.learningrate, beta1=0.5, beta2=0.9).minimize(gen_loss, var_list=gen_vars)
            optim_vae = tf.train.AdamOptimizer(learning_rate=self.config.learningrate, beta1=0.5, beta2=0.9).minimize(enc_loss, var_list=enc_vars + gen_vars)

        # initialize all variables
        tf.global_variables_initializer().run(session=self.sess)

        best_cost = inf
        last_improvement = 0
        last_epoch = self.load_checkpoint()

        # Go go go!
        for epoch in range(last_epoch, self.config.numEpochs):
            #################
            # TRAINING WGAN #
            #################
            phase = Phase.TRAIN
            scalars = defaultdict(list)
            visuals = []
            d_iters = 5
            num_batches = dataset.num_batches(self.config.batchsize, set=phase.value)
            for idx in range(0, num_batches):
                batch, _, _ = dataset.next_batch(self.config.batchsize, set=phase.value)

                # Encoder optimization
                fetches = {
                    # 'generated': self.generated,
                    'reconstruction': self.reconstruction,
                    'reconstructionLoss': self.losses['reconstructionLoss'],
                    'L1': self.losses['L1'],
                    'enc_loss': self.losses['enc_loss'],
                    'optimizer_e': optim_vae,
                }

                feed_dict = {
                    self.x: batch,
                    self.dropout: phase == Phase.TRAIN,
                    self.dropout_rate: self.config.dropout_rate
                }
                run = self.sess.run(fetches, feed_dict=feed_dict)

                # Generator optimization
                fetches = {
                    'gen_loss': self.losses['gen_loss'],
                    'optimizer_g': optim_gen,
                }

                feed_dict = {
                    self.x: batch,
                    self.dropout: phase == Phase.TRAIN,
                    self.dropout_rate: self.config.dropout_rate
                }
                run = {**run, **self.sess.run(fetches, feed_dict=feed_dict)}

                for _ in range(0, d_iters):
                    # Discriminator optimization
                    fetches = {
                        'disc_loss': self.losses['disc_loss'],
                        'disc_fake': self.losses['disc_fake'],
                        'disc_real': self.losses['disc_real'],
                        'optimizer_d': optim_dis,
                    }
                    feed_dict = {
                        self.x: batch,
                        self.dropout: phase == Phase.TRAIN,
                        self.dropout_rate: self.config.dropout_rate
                    }
                    run = {**run, **self.sess.run(fetches, feed_dict=feed_dict)}

                # Print to console
                print(f'Epoch ({phase.value}): [{epoch:2d}] [{idx:4d}/{num_batches:4d}]'
                      f' gen_loss: {run["gen_loss"]:.8f}, disc_loss: {run["disc_loss"]:.8f}, reconstructionLoss: {run["reconstructionLoss"]:.8f}')
                update_log_dicts(*trainer_utils.get_summary_dict(batch, run), scalars, visuals)

            self.log_to_tensorboard(epoch, scalars, visuals, phase)

            # Increment last_epoch counter and save model
            last_epoch += 1
            self.save(self.checkpointDir, last_epoch)

            ##############
            # VALIDATION #
            ##############
            phase = Phase.VAL
            scalars = defaultdict(list)
            visuals = []
            num_batches = dataset.num_batches(self.config.batchsize, set=phase.value)
            for idx in range(0, num_batches):
                batch, _, _ = dataset.next_batch(self.config.batchsize, set=phase.value)

                # Encoder optimization
                fetches = {
                    'reconstruction': self.reconstruction,
                    'reconstructionLoss': self.losses['reconstructionLoss'],
                    'L1': self.losses['L1'],
                    'enc_loss': self.losses['enc_loss'],
                }

                feed_dict = {
                    self.x: batch,
                    self.dropout: phase == Phase.TRAIN,
                    self.dropout_rate: self.config.dropout_rate
                }
                run = self.sess.run(fetches, feed_dict=feed_dict)
                # Print to console
                print(f'Epoch ({phase.value}): [{epoch:2d}] [{idx:4d}/{num_batches:4d}] reconstructionLoss: {run["reconstructionLoss"]:.8f}')
                update_log_dicts(*trainer_utils.get_summary_dict(batch, run), scalars, visuals)

            self.log_to_tensorboard(epoch, scalars, visuals, phase)

            best_cost, last_improvement, stop = indicate_early_stopping(scalars['reconstructionLoss'], best_cost, last_improvement)
            if stop:
                print('Early stopping was triggered due to no improvement over the last 5 epochs')
                break

    def reconstruct(self, x, dropout=False):
        if x.ndim < 4:
            x = np.expand_dims(x, 0)

        fetches = {
            'reconstruction': self.reconstruction
        }
        feed_dict = {
            self.x: x,
            self.dropout: dropout,  # apply only during MC sampling.
            self.dropout_rate: self.config.dropout_rate
        }
        results = self.sess.run(fetches, feed_dict=feed_dict)

        results['l1err'] = np.sum(np.abs(x - results['reconstruction']))
        results['l2err'] = np.sum(np.sqrt((x - results['reconstruction']) ** 2))

        return results
