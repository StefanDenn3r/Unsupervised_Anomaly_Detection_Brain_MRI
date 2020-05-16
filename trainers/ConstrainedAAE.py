from collections import defaultdict
from math import inf

from tensorflow.python.ops.losses.losses_impl import Reduction

from trainers import trainer_utils
from trainers.AEMODEL import AEMODEL, Phase, indicate_early_stopping, update_log_dicts
from trainers.DLMODEL import *


class ConstrainedAAE(AEMODEL):
    class Config(AEMODEL.Config):
        def __init__(self):
            super().__init__('ConstrainedAAE')
            self.rho = 1
            self.scale = 10.0

    def __init__(self, sess, config, network=None):
        super().__init__(sess, config, network)
        self.x = tf.placeholder(tf.float32, [None, self.config.outputHeight, self.config.outputWidth, self.config.numChannels], name='x')
        self.z = tf.placeholder(tf.float32, [None, self.config.zDim], name='z')

        self.outputs = self.network(self.z, self.x, dropout_rate=self.dropout_rate, dropout=self.dropout, config=self.config)
        self.reconstruction = self.x_hat = self.outputs['x_hat']
        self.generated = self.z_ = self.outputs['z_']
        self.d_ = self.outputs['d_']
        self.d = self.outputs['d']
        self.z_hat = self.outputs['z_hat']
        self.d_hat = self.outputs['d_hat']
        self.z_rec = self.outputs['z_rec']

        self.scale = self.config.scale
        self.rho = self.config.rho

        # Print Stats
        self.get_number_of_trainable_params()
        # Instantiate Saver
        self.saver = tf.train.Saver()

    def train(self, dataset):
        # Determine trainable variables
        self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)

        # Build losses
        self.losses['gen_loss'] = gen_loss = -tf.reduce_mean(self.d_)
        self.losses['disc_loss_without_grad'] = disc_loss = tf.reduce_mean(self.d_) - tf.reduce_mean(self.d)
        self.losses['disc_loss_real'] = tf.reduce_mean(self.d)
        self.losses['disc_loss_fake'] = tf.reduce_mean(self.d_)

        ddx = tf.gradients(self.d_hat, self.z_hat)[0]
        ddx = tf.sqrt(tf.reduce_sum(tf.square(ddx), axis=1))
        ddx = tf.reduce_mean(tf.square(ddx - 1.0) * self.scale)
        self.losses['disc_loss'] = disc_loss = disc_loss + ddx

        self.losses['L1'] = tf.losses.absolute_difference(self.x, self.reconstruction, reduction=Reduction.NONE)
        self.losses['reconstructionLoss'] = tf.reduce_mean(tf.reduce_sum(self.losses['L1'], axis=[1, 2, 3]))

        self.losses['L2'] = l2 = tf.reduce_mean(tf.losses.mean_squared_error(self.x, self.reconstruction, reduction=Reduction.NONE), axis=[1, 2, 3])
        self.losses['Rec_z'] = rec_z = tf.reduce_mean(tf.losses.mean_squared_error(self.z_rec, self.z_, reduction=Reduction.NONE), axis=[1])

        self.losses['loss'] = ae_loss = tf.reduce_mean(l2 + self.rho * rec_z)

        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            # Set the optimizer
            t_vars = tf.trainable_variables()
            dis_vars = [var for var in t_vars if 'Discriminator' in var.name]
            gen_vars = [var for var in t_vars if 'Encoder' in var.name]
            ae_vars = t_vars

            optim_dis = tf.train.AdamOptimizer(learning_rate=self.config.learningrate, beta1=0.5, beta2=0.9).minimize(disc_loss, var_list=dis_vars)
            optim_gen = tf.train.AdamOptimizer(learning_rate=self.config.learningrate, beta1=0.5, beta2=0.9).minimize(gen_loss, var_list=gen_vars)
            optim_ae = tf.train.AdamOptimizer(learning_rate=self.config.learningrate, beta1=0.5, beta2=0.9).minimize(ae_loss, var_list=ae_vars)

        # initialize all variables
        tf.global_variables_initializer().run(session=self.sess)

        best_cost = inf
        last_improvement = 0
        last_epoch = self.load_checkpoint()

        # Go go go!
        for epoch in range(last_epoch, self.config.numEpochs):
            ############
            # TRAINING #
            ############
            phase = Phase.TRAIN
            scalars = defaultdict(list)
            visuals = []
            d_iters = 20
            num_batches = dataset.num_batches(self.config.batchsize, set=phase.value)
            for idx in range(0, num_batches):
                batch, _, _ = dataset.next_batch(self.config.batchsize, set=phase.value)

                run = {}
                for _ in range(d_iters if epoch <= 5 else 1):
                    # AE optimization
                    fetches = {
                        'reconstruction': self.reconstruction,
                        'rec_z': self.losses['Rec_z'],
                        'L1': self.losses['L1'],
                        'loss': self.losses['loss'],
                        'reconstructionLoss': self.losses['reconstructionLoss'],
                        'z_': self.z_,
                        'z_rec': self.z_rec,
                        'optimizer_ae': optim_ae
                    }

                    feed_dict = self.get_feed_dict(batch, phase)

                    run = self.sess.run(fetches, feed_dict=feed_dict)

                for _ in range(d_iters):
                    # Discriminator optimization
                    fetches = {
                        'disc_loss': self.losses['disc_loss'],
                        'optimizer_d': optim_dis,
                    }

                    feed_dict = self.get_feed_dict(batch, phase)

                    run = {**run, **self.sess.run(fetches, feed_dict=feed_dict)}

                # Generator optimization
                fetches = {
                    'gen_loss': self.losses['gen_loss'],
                    'optimizer_g': optim_gen,
                }

                feed_dict = self.get_feed_dict(batch, phase)

                run = {**run, **self.sess.run(fetches, feed_dict=feed_dict)}

                # Print to console
                print(f'Epoch ({phase.value}): [{epoch:2d}] [{idx:4d}/{num_batches:4d}] loss: {run["reconstructionLoss"]:.8f},'
                      f' gen_loss: {run["gen_loss"]:.8f}, disc_loss: {run["disc_loss"]:.8f}')
                update_log_dicts(*trainer_utils.get_summary_dict(batch, run), scalars, visuals)

            self.log_to_tensorboard(epoch, scalars, visuals, phase)

            # Increment last_epoch counter and save model
            last_epoch += 1
            self.save(self.checkpointDir, last_epoch)

            ##############
            # VALIDATION #
            ##############
            scalars = defaultdict(list)
            visuals = []
            phase = Phase.VAL
            num_batches = dataset.num_batches(self.config.batchsize, set=phase.value)
            for idx in range(0, num_batches):
                batch, _, _ = dataset.next_batch(self.config.batchsize, set=phase.value)

                fetches = {
                    'reconstruction': self.reconstruction,
                    **self.losses
                }

                feed_dict = self.get_feed_dict(batch, phase)
                run = self.sess.run(fetches, feed_dict=feed_dict)

                # Print to console
                print(f'Epoch ({phase.value}): [{epoch:2d}] [{idx:4d}/{num_batches:4d}] loss: {run["loss"]:.8f}')
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
            self.z: self.sample_z(),
            self.dropout: dropout,  # apply only during MC sampling.
            self.dropout_rate: self.config.dropout_rate
        }
        results = self.sess.run(fetches, feed_dict=feed_dict)

        results['l1err'] = np.sum(np.abs(x - results['reconstruction']))
        results['l2err'] = np.sum(np.sqrt((x - results['reconstruction']) ** 2))

        return results

    def sample_z(self):
        return np.random.normal(size=[self.config.batchsize, self.config.zDim])
