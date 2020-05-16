import random
from collections import defaultdict
from math import inf

from tensorflow.python.ops.losses.losses_impl import Reduction

from trainers import trainer_utils
from trainers.AEMODEL import AEMODEL, Phase, indicate_early_stopping, update_log_dicts
from trainers.DLMODEL import *


class CE(AEMODEL):
    class Config(AEMODEL.Config):
        def __init__(self):
            super().__init__('CE')

    def __init__(self, sess, config, network=None):
        super().__init__(sess, config, network)
        self.x = tf.placeholder(tf.float32, [None, self.config.outputHeight, self.config.outputWidth, self.config.numChannels], name='x')
        self.x_ce = tf.placeholder(tf.float32, [None, self.config.outputHeight, self.config.outputWidth, self.config.numChannels], name='input_ce')
        self.outputs = self.network(self.x_ce, dropout_rate=self.dropout_rate, dropout=self.dropout, config=self.config)
        self.reconstruction = self.outputs['x_hat']

        # Print Stats
        self.get_number_of_trainable_params()
        # Instantiate Saver
        self.saver = tf.train.Saver()

    def train(self, dataset):
        # Determine trainable variables
        self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)

        # Build losses
        self.losses['L1'] = tf.losses.absolute_difference(self.x, self.reconstruction, reduction=Reduction.NONE)
        self.losses['loss'] = self.losses['reconstructionLoss'] = tf.reduce_mean(tf.reduce_sum(self.losses['L1'], axis=[1, 2, 3]))

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
            batch, _, brainmasks = dataset.next_batch(self.config.batchsize, return_brainmask=True, set=phase.value)

            masked_batch = retrieve_masked_batch(batch, brainmasks)

            fetches = {
                'reconstruction': self.reconstruction,
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
            self.x_ce: x,
            self.dropout: dropout,
            self.dropout_rate: self.config.dropout_rate
        }
        results = self.sess.run(fetches, feed_dict=feed_dict)

        results['l1err'] = np.sum(np.abs(x - results['reconstruction']))
        results['l2err'] = np.sum(np.sqrt((x - results['reconstruction']) ** 2))

        return results


def retrieve_masked_batch(batch, brainmasks):
    def retrieve_brain_range(brainmask):
        pixels = np.argwhere(brainmask).T
        return (min(pixels[0]), max(pixels[0])), (min(pixels[1]), max(pixels[1]))

    brain_ranges = list(map(lambda brainmask: retrieve_brain_range(brainmask), brainmasks))
    # Masking out for Context Encoder
    m = np.ones(batch.shape)
    for (m, brain_range) in zip(m, brain_ranges):
        for _ in range(random.randint(1, 3)):
            size_w, size_h = 20, 20
            if brain_range[0][0] < brain_range[0][1] - size_w and brain_range[1][0] < brain_range[1][1] - size_h:
                x = random.randint(brain_range[0][0], brain_range[0][1] - size_w)
                y = random.randint(brain_range[1][0], brain_range[1][1] - size_h)
                m[x:x + size_w, y:y + size_h] = 0
    masked_batch = batch * m
    return masked_batch
