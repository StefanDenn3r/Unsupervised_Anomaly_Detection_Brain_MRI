import os
from abc import ABC
from datetime import datetime

import numpy as np
import tensorflow as tf

from trainers.DLMODEL import DLMODEL
from utils.logger import Logger, Phase


class AEMODEL(DLMODEL, ABC):
    class Config(DLMODEL.Config):
        def __init__(self, modelname='AE'):
            super().__init__()
            self.modelname = modelname
            self.intermediateResolutions = [8, 8]
            self.outputWidth = 256
            self.outputHeight = 256
            self.numChannels = 3
            self.dropout = False
            self.dropout_rate = 0.2
            self.zDim = 128

    def __init__(self, sess, config=Config(), network=None):
        super().__init__(sess, config)
        self.losses = {}
        self.dropout = tf.placeholder(tf.bool, name='dropout')
        self.dropout_rate = tf.placeholder(tf.float32, name='dropout_rate')

        self.network = network
        self.checkpointDir = os.path.join(self.config.checkpointDir, self.network.__name__)
        self.logDir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'logs', self.network.__name__, self.model_dir,
                                   datetime.now().strftime('%Y%m%d_%H%M%S'))
        self.logger = Logger(self.sess, self.logDir)

    def log_to_tensorboard(self, epoch, scalars, visuals, phase: Phase, name='x'):
        for key in scalars.keys():
            scalars[key] = np.mean(scalars[key])

        if visuals:
            self.logger.summarize(epoch, phase=phase, summaries_dict={**scalars, **{name: np.vstack(visuals)[:50]}})

    def load_checkpoint(self):
        could_load, checkpoint_counter = self.load(self.checkpointDir)
        if could_load:
            last_epoch = checkpoint_counter
            print(" [*] Load SUCCESS")
        else:
            last_epoch = 0
            print(" [!] Load failed...")
        return last_epoch

    @property
    def model_dir(self):
        return "{}_d{}_s{}x{}_{}_b{}_z{}_{}".format(self.config.modelname, self.config.dataset,
                                                    self.config.outputWidth,
                                                    self.config.outputHeight,
                                                    self.network.__name__,
                                                    self.config.batchsize, self.config.zDim,
                                                    self.config.description)


def update_log_dicts(scalars, visuals, train_scalars, train_visuals):
    for k, v in list(scalars.items()):
        train_scalars[k].append(v)
    train_visuals.append(visuals)


def indicate_early_stopping(current_cost, best_cost, last_improvement):
    if current_cost < best_cost:
        best_cost = current_cost
        last_improvement = 0
        return best_cost, last_improvement, False
    else:
        last_improvement += 1
        if last_improvement >= 5:
            return best_cost, last_improvement, True
        return best_cost, last_improvement, False
