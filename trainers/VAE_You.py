from collections import defaultdict
from math import inf

from tensorflow.python.ops.losses.losses_impl import Reduction

from trainers import trainer_utils
from trainers.AEMODEL import Phase, update_log_dicts, indicate_early_stopping, AEMODEL
from trainers.DLMODEL import *


class VAE_You(AEMODEL):
    class Config(AEMODEL.Config):
        def __init__(self):
            super().__init__('VAE_You')
            self.restore_lr = 1e-3
            self.restore_steps = 150
            self.tv_lambda = 1.8

    def __init__(self, sess, config, network=None):
        super().__init__(sess, config, network)
        self.x = tf.placeholder(tf.float32, [None, self.config.outputHeight, self.config.outputWidth, self.config.numChannels], name='x')
        self.tv_lambda = tf.placeholder(tf.float32, shape=())

        # Additional Parameters
        self.restore_lr = self.config.restore_lr
        self.restore_steps = self.config.restore_steps
        self.tv_lambda_value = self.config.tv_lambda

        self.outputs = self.network(self.x, dropout_rate=self.dropout_rate, dropout=self.dropout, config=self.config)
        self.reconstruction = self.outputs['x_hat']
        self.z_mu = self.outputs['z_mu']
        self.z_sigma = self.outputs['z_sigma']

        # Print Stats
        self.get_number_of_trainable_params()
        # Instantiate Saver
        self.saver = tf.train.Saver()

    def train(self, dataset):
        # Determine trainable variables
        self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)

        # Build losses
        self.losses['L1'] = tf.losses.absolute_difference(self.x, self.reconstruction, reduction=Reduction.NONE)
        rec = tf.reduce_sum(self.losses['L1'], axis=[1, 2, 3])
        kl = 0.5 * tf.reduce_sum(tf.square(self.z_mu) + tf.square(self.z_sigma) - tf.log(tf.square(self.z_sigma)) - 1, axis=1)
        self.losses['pixel_loss'] = rec + kl
        self.losses['reconstructionLoss'] = tf.reduce_mean(rec)
        self.losses['kl'] = tf.reduce_mean(kl)
        self.losses['loss'] = tf.reduce_mean(rec + kl)

        # for restoration
        self.losses['restore'] = self.tv_lambda * tf.image.total_variation(tf.subtract(self.x, self.reconstruction))
        self.losses['grads'] = tf.gradients(self.losses['pixel_loss'] + self.losses['restore'], self.x)[0]

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

        if self.tv_lambda_value == -1 and self.restore_steps > 0:
            ##############
            # Determine lambda #
            ##############
            print('Determining best lambda')
            self.determine_best_lambda(dataset)

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
                self.tv_lambda: self.tv_lambda_value,
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

        restored = x.copy()
        for step in range(self.restore_steps):
            feed_dict = {
                self.x: restored,
                self.tv_lambda: self.tv_lambda_value,
                self.dropout: dropout,  # apply only during MC sampling.
                self.dropout_rate: self.config.dropout_rate
            }
            run = self.sess.run({'grads': self.losses['grads']}, feed_dict=feed_dict)
            gradients = run['grads']
            restored -= self.restore_lr * gradients

        results = {
            'reconstruction': restored
        }
        results['l1err'] = np.sum(np.abs(x - results['reconstruction']))
        results['l2err'] = np.sum(np.sqrt((x - results['reconstruction']) ** 2))

        return results

    def determine_best_lambda(self, dataset):
        lambdas = np.arange(20) / 10.0
        mean_errors = []
        fetches = self.losses

        for tv_lambda in lambdas:
            errors = []
            for idx in range(int(dataset.num_batches(self.config.batchsize, set=Phase.VAL.value) * 0.2)):
                batch, _, _ = dataset.next_batch(self.config.batchsize, set=Phase.VAL.value)
                restored = batch.copy()
                for step in range(self.restore_steps):
                    feed_dict = {
                        self.x: restored,
                        self.tv_lambda: tv_lambda,
                        self.dropout: False,
                        self.dropout_rate: self.config.dropout_rate
                    }
                    run = self.sess.run(fetches, feed_dict=feed_dict)
                    restored -= self.restore_lr * run['grads']
                errors.append(np.sum(np.abs(batch - restored)))
            mean_error = np.mean(errors)
            mean_errors.append(mean_error)
            print(f'mean_error for lambda {tv_lambda}: {mean_error}')
        self.tv_lambda_value = lambdas[mean_errors.index(min(mean_errors))]
        print(f'Best lambda: {self.tv_lambda_value}')
