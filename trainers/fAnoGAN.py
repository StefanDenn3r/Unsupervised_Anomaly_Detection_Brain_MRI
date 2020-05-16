from collections import defaultdict
from math import inf

from tensorflow.python.ops.losses.losses_impl import Reduction

from trainers import trainer_utils
from trainers.AEMODEL import AEMODEL, Phase, indicate_early_stopping, update_log_dicts
from trainers.DLMODEL import *


class fAnoGAN(AEMODEL):
    class Config(AEMODEL.Config):
        def __init__(self):
            super().__init__('fAnoGAN')
            self.scale = 10.0
            self.kappa = 1.0

    def __init__(self, sess, config, network=None):
        super().__init__(sess, config, network)
        self.x = tf.placeholder(tf.float32, [None, self.config.outputHeight, self.config.outputWidth, self.config.numChannels], name='x')
        self.z = tf.placeholder(tf.float32, [None, self.config.zDim], name='z')

        self.outputs = self.network(self.z, self.x, dropout_rate=self.dropout_rate, dropout=self.dropout, config=self.config)
        self.z_enc = self.outputs['z_enc']
        self.generated = self.x_ = self.outputs['x_']
        self.reconstruction = self.x_enc = self.outputs['x_enc']
        self.d_fake_features = self.outputs['d_fake_features']
        self.d_ = self.outputs['d_']
        self.d_features = self.outputs['d_features']
        self.d = self.outputs['d']
        self.x_hat = self.outputs['x_hat']
        self.d_hat_features = self.outputs['d_hat_features']
        self.d_hat = self.outputs['d_hat']
        self.d_enc_features = self.outputs['d_enc_features']
        self.d_enc = self.outputs['d_enc']

        self.kappa = self.config.kappa
        self.scale = self.config.scale

        # Print Stats
        self.get_number_of_trainable_params()
        # Instantiate Saver
        self.saver = tf.train.Saver()

    def train(self, dataset):
        # Determine trainable variables
        self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)

        # Build losses
        self.losses['disc_real'] = disc_real = tf.reduce_mean(self.d)
        self.losses['disc_fake'] = disc_fake = tf.reduce_mean(self.d_)
        self.losses['gen_loss'] = gen_loss = -disc_fake
        disc_loss = disc_fake - disc_real

        ddx = tf.gradients(self.d_hat, self.x_hat)[0]  # gradient
        ddx = tf.sqrt(tf.reduce_sum(tf.square(ddx), axis=1))  # slopes
        ddx = tf.reduce_mean(tf.square(ddx - 1.0)) * self.scale  # gradient penalty
        self.losses['disc_loss'] = disc_loss = disc_loss + ddx

        self.losses['loss_img'] = loss_img = tf.reduce_mean(
            tf.reduce_mean(tf.losses.mean_squared_error(self.x, self.x_enc, reduction=Reduction.NONE), axis=[1, 2, 3]))
        self.losses['loss_fts'] = loss_fts = tf.reduce_mean(
            tf.reduce_mean(tf.losses.mean_squared_error(self.d_enc_features, self.d_features, reduction=Reduction.NONE), axis=[1, 2, 3]))
        self.losses['enc_loss'] = enc_loss = loss_img + self.kappa * loss_fts
        self.losses['L1'] = tf.losses.absolute_difference(self.x, self.x_enc, reduction=Reduction.NONE)
        self.losses['reconstructionLoss'] = self.losses['loss'] = tf.reduce_mean(tf.reduce_sum(self.losses['L1'], axis=[1, 2, 3]))

        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            # Set the optimizer
            t_vars = tf.trainable_variables()
            dis_vars = [var for var in t_vars if 'Discriminator' in var.name]
            gen_vars = [var for var in t_vars if 'Generator' in var.name]
            enc_vars = [var for var in t_vars if 'Encoder' in var.name]

            optim_dis = tf.train.AdamOptimizer(learning_rate=self.config.learningrate, beta1=0.5, beta2=0.9).minimize(disc_loss, var_list=dis_vars)
            optim_gen = tf.train.AdamOptimizer(learning_rate=self.config.learningrate, beta1=0.5, beta2=0.9).minimize(gen_loss, var_list=gen_vars)
            optim_enc = tf.train.AdamOptimizer(learning_rate=self.config.learningrate, beta1=0.5, beta2=0.9).minimize(enc_loss, var_list=enc_vars)

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

                # Generator optimization
                fetches = {
                    'generated': self.generated,
                    'gen_loss': self.losses['gen_loss'],
                    'optimizer_g': optim_gen,
                }

                feed_dict = {
                    self.x: batch,
                    self.z: self.sample_z(),
                    self.dropout: phase == Phase.TRAIN,
                    self.dropout_rate: self.config.dropout_rate
                }
                run = self.sess.run(fetches, feed_dict=feed_dict)

                for _ in range(0, d_iters):
                    # Discriminator optimization
                    fetches = {
                        'generated': self.generated,
                        'disc_loss': self.losses['disc_loss'],
                        'disc_fake': self.losses['disc_fake'],
                        'disc_real': self.losses['disc_real'],
                        'optimizer_d': optim_dis,
                    }
                    feed_dict = {
                        self.x: batch,
                        self.z: self.sample_z(),
                        self.dropout: phase == Phase.TRAIN,
                        self.dropout_rate: self.config.dropout_rate
                    }
                    run = {**run, **self.sess.run(fetches, feed_dict=feed_dict)}

                # Print to console
                print(f'Epoch ({phase.value} WGAN): [{epoch:2d}] [{idx:4d}/{num_batches:4d}]'
                      f' gen_loss: {run["gen_loss"]:.8f}, disc_loss: {run["disc_loss"]:.8f}')
                update_log_dicts(*trainer_utils.get_summary_dict(batch, run, visualization_keys=['generated']), scalars, visuals)

            self.log_to_tensorboard(epoch, scalars, visuals, phase, name='wgan_x')

            # Increment last_epoch counter and save model
            last_epoch += 1
            self.save(self.checkpointDir, last_epoch)

        for epoch in range(last_epoch, 2 * self.config.numEpochs):
            ####################
            # TRAINING Encoder #
            ####################
            phase = Phase.TRAIN
            scalars = defaultdict(list)
            visuals = []
            num_batches = dataset.num_batches(self.config.batchsize, set=phase.value)
            for idx in range(0, num_batches):
                batch, _, _ = dataset.next_batch(self.config.batchsize, set=phase.value)
                fetches = {
                    'reconstruction': self.reconstruction,
                    'optimizer_enc': optim_enc,
                    'z_enc': self.z_enc,
                    'z': self.z,
                    **self.losses
                }

                feed_dict = {
                    self.x: batch,
                    self.z: self.sample_z(),
                    self.dropout: phase == Phase.TRAIN,
                    self.dropout_rate: self.config.dropout_rate
                }
                run = self.sess.run(fetches, feed_dict=feed_dict)

                # Print to console
                print(f'Epoch ({phase.value} Encoder): [{epoch:2d}] [{idx:4d}/{num_batches:4d}]  reconstructionLoss: {run["reconstructionLoss"]:.8f}')
                update_log_dicts(*trainer_utils.get_summary_dict(batch, run), scalars, visuals)

            self.log_to_tensorboard(epoch, scalars, visuals, phase)

            # Increment last_epoch counter and save model
            last_epoch += 1
            self.save(self.checkpointDir, last_epoch)

            ######################
            # VALIDATION Encoder #
            ######################
            phase = Phase.VAL
            scalars = defaultdict(list)
            visuals = []
            num_batches = dataset.num_batches(self.config.batchsize, set=phase.value)
            for idx in range(0, num_batches):
                batch, _, _ = dataset.next_batch(self.config.batchsize, set=phase.value)

                fetches = {
                    'reconstruction': self.reconstruction,
                    **self.losses
                }

                feed_dict = {
                    self.x: batch,
                    self.z: self.sample_z(),
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

    def get_feed_dict(self, batch, phase):
        return {
            self.x: batch,
            self.z: self.sample_z(),
            self.dropout: phase == Phase.TRAIN,
            self.dropout_rate: self.config.dropout_rate
        }

    def reconstruct(self, x, dropout=False):
        if x.ndim < 4:
            x = np.expand_dims(x, 0)

        fetches = {
            'reconstruction': self.reconstruction
        }

        feed_dict = {
            self.x: x,
            self.z: self.sample_z(x.shape[0]),
            self.dropout: dropout,  # apply only during MC sampling.
            self.dropout_rate: self.config.dropout_rate
        }
        results = self.sess.run(fetches, feed_dict=feed_dict)

        results['l1err'] = np.sum(np.abs(x - results['reconstruction']))
        results['l2err'] = np.sum(np.sqrt((x - results['reconstruction']) ** 2))

        return results

    def sample_z(self, batch_size=None):
        return np.random.normal(size=[batch_size if batch_size else self.config.batchsize, self.config.zDim])
