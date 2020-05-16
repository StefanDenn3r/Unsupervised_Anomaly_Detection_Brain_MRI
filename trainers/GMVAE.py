from collections import defaultdict
from math import inf

from tensorflow.python.ops.losses.losses_impl import Reduction

from trainers import trainer_utils
from trainers.AEMODEL import Phase, update_log_dicts, indicate_early_stopping, AEMODEL
from trainers.DLMODEL import *


class GMVAE(AEMODEL):
    class Config(AEMODEL.Config):
        def __init__(self):
            super().__init__('GMVAE')
            self.dim_c = 6
            self.dim_z = 1
            self.dim_w = 1
            self.c_lambda = 1
            self.restore_lr = 1e-3
            self.restore_steps = 150
            self.tv_lambda = 1.8

    def __init__(self, sess, config, network=None):
        super().__init__(sess, config, network)
        self.x = tf.placeholder(tf.float32, [None, self.config.outputHeight, self.config.outputWidth, self.config.numChannels], name='x')
        self.tv_lambda = tf.placeholder(tf.float32, shape=())

        # Additional Parameters
        self.dim_c = self.config.dim_c
        self.dim_z = self.config.dim_z
        self.dim_w = self.config.dim_w
        self.c_lambda = self.config.c_lambda
        self.restore_lr = self.config.restore_lr
        self.restore_steps = self.config.restore_steps
        self.tv_lambda_value = self.config.tv_lambda

        self.outputs = self.network(self.x, dropout_rate=self.dropout_rate, dropout=self.dropout, config=self.config)

        self.w_mu = self.outputs['w_mu']
        self.w_log_sigma = self.outputs['w_log_sigma']
        self.z_sampled = self.outputs['z_sampled']
        self.z_mu = self.outputs['z_mu']
        self.z_log_sigma = self.outputs['z_log_sigma']
        self.z_wc_mu = self.outputs['z_wc_mus']
        self.z_wc_log_sigma_inv = self.outputs['z_wc_log_sigma_invs']
        self.xz_mu = self.outputs['xz_mu']
        self.pc = self.outputs['pc']
        self.reconstruction = self.xz_mu

        # Print Stats
        self.get_number_of_trainable_params()
        # Instantiate Saver
        self.saver = tf.train.Saver()

    def train(self, dataset):
        # Determine trainable variables
        self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)

        # Build losses
        # 1. the reconstruction loss
        self.losses['L1'] = tf.losses.absolute_difference(self.x, self.xz_mu, reduction=Reduction.NONE)
        self.losses['L1_sum'] = tf.reduce_sum(self.losses['L1'], axis=[1, 2, 3])
        self.losses['reconstructionLoss'] = self.losses['mean_p_loss'] = mean_p_loss = tf.reduce_mean(self.losses['L1_sum'])
        self.losses['L2'] = tf.losses.mean_squared_error(self.x, self.xz_mu, reduction=Reduction.NONE)
        self.losses['L2_sum'] = tf.reduce_sum(self.losses['L2'])

        # 2. E_c_w[KL(q(z|x)|| p(z|w, c))]
        # calculate KL for each cluster
        # KL  = 1/2(  logvar2 - logvar1 + (var1 + (m1-m2)^2)/var2  - 1 ) here dim_c clusters, then we have batchsize * dim_z * dim_c
        # then [batchsize * dim_z* dim_c] * [batchsize * dim_c * 1]  = batchsize * dim_z * 1, squeeze it to batchsize * dim_z
        self.z_mu = tf.tile(tf.expand_dims(self.z_mu, -1), [1, 1, self.dim_c])
        z_logvar = tf.tile(tf.expand_dims(self.z_log_sigma, -1), [1, 1, self.dim_c])
        d_mu_2 = tf.squared_difference(self.z_mu, self.z_wc_mu)
        d_var = (tf.exp(z_logvar) + d_mu_2) * (tf.exp(self.z_wc_log_sigma_inv) + 1e-6)
        d_logvar = -1 * (self.z_wc_log_sigma_inv + z_logvar)
        kl = (d_var + d_logvar - 1) * 0.5
        con_prior_loss = tf.reduce_sum(tf.squeeze(tf.matmul(kl, tf.expand_dims(self.pc, -1)), -1), 1)
        self.losses['conditional_prior_loss'] = mean_con_loss = tf.reduce_mean(con_prior_loss)

        # 3. KL(q(w|x)|| p(w) ~ N(0, I))
        # KL = 1/2 sum( mu^2 + var - logvar -1 )
        w_loss = 0.5 * tf.reduce_sum(tf.square(self.w_mu) + tf.exp(self.w_log_sigma) - self.w_log_sigma - 1, 1)
        self.losses['w_prior_loss'] = mean_w_loss = tf.reduce_mean(w_loss)

        # 4. KL(q(c|z)||p(c)) =  - sum_k q(k) log p(k)/q(k) , k = dim_c
        # let p(k) = 1/K#

        closs1 = tf.reduce_sum(tf.multiply(self.pc, tf.log(self.pc * self.dim_c + 1e-8)), [1])
        c_lambda = tf.cast(tf.fill(tf.shape(closs1), self.c_lambda), dtype=tf.float32)
        c_loss = tf.maximum(closs1, c_lambda)
        self.losses['c_prior_loss'] = mean_c_loss = tf.reduce_mean(c_loss)

        self.losses['loss'] = mean_p_loss + mean_con_loss + mean_w_loss + mean_c_loss
        self.losses['restore'] = self.tv_lambda * tf.image.total_variation(tf.subtract(self.x, self.reconstruction))
        self.losses['grads'] = tf.gradients(self.losses['loss'] + self.losses['restore'], self.x)[0]

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
            self.process(dataset, epoch, Phase.TRAIN, optim, visualization_keys=['reconstruction', 'L1', 'L2'])

            # Increment last_epoch counter and save model
            last_epoch += 1
            self.save(self.checkpointDir, last_epoch)

            ##############
            # VALIDATION #
            ##############
            val_scalars = self.process(dataset, epoch, Phase.VAL, visualization_keys=['reconstruction', 'L1', 'L2'])

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

    def process(self, dataset, epoch, phase: Phase, optim=None, visualization_keys=None):
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
            update_log_dicts(*trainer_utils.get_summary_dict(batch, run, visualization_keys), scalars, visuals)

        self.log_to_tensorboard(epoch, scalars, visuals, phase)
        return scalars

    def reconstruct(self, x, dropout=False):
        if x.ndim < 4:
            x = np.expand_dims(x, 0)

        if self.restore_steps == 0:
            feed_dict = {
                self.x: x,
                self.tv_lambda: self.tv_lambda_value,
                self.dropout: dropout,
                self.dropout_rate: self.config.dropout_rate
            }
            results = self.sess.run({'reconstruction': self.reconstruction}, feed_dict=feed_dict)
        else:
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
