import json
import os
from abc import abstractmethod

import matplotlib.pyplot
import numpy as np
import tensorflow as tf


# Baseline class for all your Deep Learning needs with TensorFlow #

class DLMODEL(object):
    class Config(object):
        def __init__(self):
            self.modelname = ''
            self.model_config = {}
            self.checkpointDir = None
            self.description = ''
            self.batchsize = 6
            self.useTensorboard = True
            self.tensorboardPort = 8008
            self.useMatplotlib = False
            self.debugGradients = False
            self.tfSummaryAfter = 100
            self.dataset = ''
            self.beta1 = 0.5

    def __init__(self, sess, config=Config()):
        """

        Args:
          sess: TensorFlow session
          config: (optional) a DLMODEL.Config object with your options
        """
        self.sess = sess
        self.config = config
        self.variables = {}
        self.curves = {}  # For plotting via matplotlib
        self.phase = tf.placeholder(tf.bool, name='phase')
        self.handles = {}
        if self.config.useMatplotlib:
            self.handles['curves'] = matplotlib.pyplot.figure()
            self.handles['samples'] = matplotlib.pyplot.figure()

        self.saver = None
        self.losses = None

    @abstractmethod
    def train(self, dataset):
        """Train a Deep Neural Network"""

    def initialize_variables(self):
        uninitialized_var_names_raw = set(self.sess.run(tf.report_uninitialized_variables()))
        uninitialized_var_names = [v.decode() for v in uninitialized_var_names_raw]
        variables_to_initialize = [v for v in tf.global_variables() if v.name.split(':')[0] in uninitialized_var_names]
        self.sess.run(tf.initialize_variables(variables_to_initialize))
        print("Initialized all unitialized variables.")

    @property
    def model_dir(self):
        return "{}_d{}_b{}_{}".format(self.config.modelname, self.config.dataset, self.config.batchsize, self.config.description)

    def save(self, checkpoint_dir, step):
        model_name = self.config.modelname + ".model"
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

        # Create checkpoint directory, if it does not exist
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        # Save the current model state to the checkpoint directory
        self.saver.save(self.sess,
                        os.path.join(checkpoint_dir, model_name),
                        global_step=step)

        # Save the config to a json file, such that you can inspect it later
        with open(os.path.join(checkpoint_dir, 'Config-{}.json'.format(step)), 'w') as outfile:
            try:
                json.dump(self.config.__dict__, outfile)
            except:
                print("Failed to save config json")

        # Save the curves to a np file such that we can recover and monitor the entire training process
        np.save(os.path.join(checkpoint_dir, 'Curves.npy'), self.curves)

    def load(self, checkpoint_dir, iteration=None):
        import re
        print(" [*] Reading checkpoints...")
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

        # Load training curves, if any
        curves_file = os.path.join(checkpoint_dir, 'Curves.npy')
        if os.path.isfile(curves_file):
            self.curves = np.load(curves_file, allow_pickle=True).item()

        if iteration is not None:
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, self.config.modelname + '.model-' + str(iteration)))
            counter = iteration
            return True, counter

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            counter = int(next(re.finditer('(\d+)(?!.*\d)', ckpt_name)).group(0))
            print(" [*] Success to read {}".format(ckpt_name))
            return True, counter
        else:
            print(" [*] Failed to find a checkpoint")
            return False, 0

    @staticmethod
    def create_optimizer(loss, var_list=(), learningrate=0.001, type='ADAM', beta1=0.05, momentum=0.9, name='optimizer', minimize=True, scope=None):
        if type == 'ADAM':
            optim = tf.train.AdamOptimizer(learningrate, beta1=beta1, name=name)
        elif type == 'SGD':
            optim = tf.train.GradientDescentOptimizer(learningrate)
        elif type == 'MOMENTUM':
            optim = tf.train.MomentumOptimizer(learning_rate=learningrate, momentum=momentum)
        elif type == 'RMS':
            optim = tf.train.RMSPropOptimizer(learning_rate=learningrate, momentum=momentum)
        else:
            raise ValueError('Invalid optimizer type')

        if minimize:
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                train_op = optim.minimize(loss, var_list=var_list)
            return train_op
        else:
            return optim

    @staticmethod
    def get_number_of_trainable_params():
        def inner_get_number_of_trainable_params(_scope):
            total_parameters = 0
            _variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, _scope)
            for variable in _variables:
                # shape is an array of tf.Dimension
                shape = variable.get_shape()
                variable_parametes = 1
                for dim in shape:
                    variable_parametes *= dim.value
                total_parameters += variable_parametes
            return total_parameters

        variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "")
        scopes = list(set(map(lambda variable: os.path.split(os.path.split(variable.name)[0])[0], variables)))
        for scope in scopes:
            if scope != '':
                print(f'#Params in {scope}: {inner_get_number_of_trainable_params(scope)}')
        print(f'#Params in total: {inner_get_number_of_trainable_params("")}')
