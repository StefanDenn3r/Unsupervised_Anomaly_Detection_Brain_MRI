from collections import defaultdict
from math import inf

from tensorflow.python.ops.losses.losses_impl import Reduction

from trainers import trainer_utils
from trainers.AEMODEL import AEMODEL, Phase, indicate_early_stopping, update_log_dicts
from trainers.CE import retrieve_masked_batch
from trainers.DLMODEL import *


class ceVAE(AEMODEL):
    class Config(AEMODEL.Config):
        def __init__(self):
            super().__init__('ceVAE')
            self.use_gradient_based_restoration = True

    def __init__(self, sess, config, network=None):
        super().__init__(sess, config, network)
        self.x = tf.placeholder(tf.float32, [None, self.config.outputHeight, self.config.outputWidth, self.config.numChannels], name='x')
        self.x_ce = tf.placeholder(tf.float32, [None, self.config.outputHeight, self.config.outputWidth, self.config.numChannels], name='x_ce')
        self.outputs = self.network(self.x, self.x_ce, dropout_rate=self.dropout_rate, dropout=self.dropout, config=self.config)
        self.reconstruction = self.outputs['x_hat']
        self.reconstruction_ce = self.outputs['x_hat_ce']
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
        self.losses['L1_vae'] = tf.losses.absolute_difference(self.x, self.reconstruction, reduction=Reduction.NONE)
        self.losses['L1_ce'] = tf.losses.absolute_difference(self.x_ce, self.reconstruction_ce, reduction=Reduction.NONE)
        self.losses['L1'] = 0.5 * (self.losses['L1_vae'] + self.losses['L1_ce'])
        rec_vae = tf.reduce_sum(self.losses['L1_vae'], axis=[1, 2, 3])
        rec_ce = tf.reduce_sum(self.losses['L1_ce'], axis=[1, 2, 3])
        kl = 0.5 * tf.reduce_sum(tf.square(self.z_mu) + tf.square(self.z_sigma) - tf.log(tf.square(self.z_sigma)) - 1, axis=1)

        self.losses['Rec_ce'] = tf.reduce_mean(rec_ce)
        self.losses['Rec_vae'] = tf.reduce_mean(rec_vae)
        self.losses['reconstructionLoss'] = 0.5 * tf.reduce_mean(rec_vae + rec_ce)
        self.losses['kl'] = tf.reduce_mean(kl)
        self.losses['loss'] = tf.reduce_mean(rec_vae + kl + rec_ce)
        self.losses['loss_vae'] = tf.reduce_mean(rec_vae + kl)
        self.losses['anomaly'] = self.losses['L1_vae'] * tf.abs(tf.gradients(self.losses['loss_vae'], self.x))[0]

        # Set the optimizer
        optim = self.create_optimizer(self.losses['loss'], var_list=self.variables, learningrate=self.config.learningrate,
                                      beta1=self.config.beta1, type=self.config.optimizer)

        # initialize all variables
        tf.global_variables_initializer().run(session=self.sess)

        best_cost = inf
        last_improvement = 0
        last_epoch = self.load_checkpoint()

        visualization_keys = ['reconstruction', 'reconstruction_ce', 'anomaly']
        # Go go go!
        for epoch in range(last_epoch, self.config.numEpochs):
            ############
            # TRAINING #
            ############
            self.process(dataset, epoch, Phase.TRAIN, optim, visualization_keys=visualization_keys)

            # Increment last_epoch counter and save model
            last_epoch += 1
            self.save(self.checkpointDir, last_epoch)

            ##############
            # VALIDATION #
            ##############
            val_scalars = self.process(dataset, epoch, Phase.VAL, visualization_keys=visualization_keys)

            best_cost, last_improvement, stop = indicate_early_stopping(val_scalars['loss'], best_cost, last_improvement)
            if stop:
                print('Early stopping was triggered due to no improvement over the last 5 epochs')
                break

    def process(self, dataset, epoch, phase: Phase, optim=None, visualization_keys=None):
        scalars = defaultdict(list)
        visuals = []
        num_batches = dataset.num_batches(self.config.batchsize, set=phase.value)
        for idx in range(0, num_batches):
            batch, _, brainmasks = dataset.next_batch(self.config.batchsize, return_brainmask=True, set=phase.value)

            masked_batch = retrieve_masked_batch(batch, brainmasks)

            fetches = {
                'reconstruction': self.reconstruction,
                'reconstruction_ce': self.reconstruction_ce,
                **self.losses
            }
            if phase == Phase.TRAIN:
                fetches['optimizer'] = optim

            feed_dict = {
                self.x: batch,
                self.x_ce: masked_batch if phase == Phase.TRAIN else batch,
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

        fetches = {
            'reconstruction': self.reconstruction,
            **self.losses
        }

        feed_dict = {
            self.x: x,
            self.x_ce: x,
            self.dropout: dropout,
            self.dropout_rate: self.config.dropout_rate
        }
        results = self.sess.run(fetches, feed_dict=feed_dict)

        if self.config.use_gradient_based_restoration:
            # this is actually not the real 'reconstruction' but for convenience we treat it like it
            # would be to prevent changes in our evaluation script
            results['reconstruction'] = x - self.config.use_gradient_based_restoration * results['anomaly']

        results['l1err'] = np.sum(np.abs(x - results['reconstruction']))
        results['l2err'] = np.sum(np.sqrt((x - results['reconstruction']) ** 2))

        return results
