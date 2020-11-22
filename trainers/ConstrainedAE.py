from collections import defaultdict
from math import inf

from tensorflow.python.ops.losses.losses_impl import Reduction

from trainers import trainer_utils
from trainers.AEMODEL import AEMODEL, Phase, indicate_early_stopping, update_log_dicts
from trainers.DLMODEL import *


class ConstrainedAE(AEMODEL):
    class Config(AEMODEL.Config):
        def __init__(self):
            super().__init__('ConstrainedAE')
            self.rho = 1

    def __init__(self, sess, config, network=None):
        super().__init__(sess, config, network)
        self.x = tf.placeholder(tf.float32, [None, self.config.outputHeight, self.config.outputWidth, self.config.numChannels], name='x')
        self.outputs = self.network(self.x, dropout_rate=self.dropout_rate, dropout=self.dropout, config=self.config)
        self.reconstruction = self.outputs['x_hat']
        self.z = self.outputs['z']
        self.z_rec = self.outputs['z_rec']

        self.rho = self.config.rho

        # Print Stats
        self.get_number_of_trainable_params()
        # Instantiate Saver
        self.saver = tf.train.Saver()

    def train(self, dataset):
        # Determine trainable variables
        self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)

        # Build losses
        self.losses['L1'] = tf.losses.absolute_difference(self.x, self.reconstruction, reduction=Reduction.NONE)
        self.losses['reconstructionLoss'] = tf.reduce_mean(tf.reduce_sum(self.losses['L1'], axis=[1, 2, 3]))

        self.losses['L2'] = l2 = tf.reduce_mean(tf.losses.mean_squared_error(self.x, self.reconstruction, reduction=Reduction.NONE), axis=[1, 2, 3])
        self.losses['Rec_z'] = rec_z = tf.reduce_mean(tf.losses.mean_squared_error(self.z, self.z_rec, reduction=Reduction.NONE), axis=[1])

        self.losses['loss'] = tf.reduce_mean(l2 + self.rho * rec_z)

        # Set the optimizer
        optim = self.create_optimizer(self.losses['loss'], var_list=self.variables, learningrate=self.config.learningrate,
                                      beta1=self.config.beta1, type=self.config.optimizer)

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
            self.process(dataset, epoch, Phase.TRAIN, optim)

            # Increment last_epoch counter and save model
            last_epoch += 1
            self.save(self.checkpointDir, last_epoch)

            ##############
            # VALIDATION #
            ##############
            val_scalars = self.process(dataset, epoch, Phase.VAL)

            best_cost, last_improvement, stop = indicate_early_stopping(val_scalars['loss'], best_cost, last_improvement)
            if stop:
                print('Early stopping was triggered due to no improvement over the last 5 epochs')
                break

    def process(self, dataset, epoch, phase: Phase, optim=None):
        scalars = defaultdict(list)
        visuals = []
        num_batches = dataset.num_batches(self.config.batchsize, set=phase.value)
        for idx in range(0, num_batches):
            batch, _, _ = dataset.next_batch(self.config.batchsize, set=phase.value)

            fetches = {
                'reconstruction': self.reconstruction,
                **self.losses
            }
            if phase == Phase.TRAIN:
                fetches['optimizer'] = optim

            feed_dict = {
                self.x: batch,
                self.dropout: phase == Phase.TRAIN,
                self.dropout_rate: self.config.dropout_rate
            }

            run = self.sess.run(fetches, feed_dict=feed_dict)

            # Print to console
            print(f'Epoch ({phase.value}): [{epoch:2d}] [{idx:4d}/{num_batches:4d}] loss: {run["loss"]:.8f}')
            update_log_dicts(*trainer_utils.get_summary_dict(batch, run), scalars, visuals)

        self.log_to_tensorboard(epoch, scalars, visuals, phase)
        return scalars

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
