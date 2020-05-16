#!/usr/bin/env python
import tensorflow as tf

from models.constrained_adversarial_autoencoder_Chen import constrained_adversarial_autoencoder_Chen
from trainers.ConstrainedAAE import ConstrainedAAE
from utils import Evaluation
from utils.default_config_setup import get_config, get_options, get_datasets, Dataset

tf.reset_default_graph()
dataset = Dataset.BRAINWEB
options = get_options(batchsize=8, learningrate=0.001, numEpochs=1, zDim=128, outputWidth=128, outputHeight=128)
options['data']['dir'] = options["globals"][dataset.value]
datasetHC, datasetPC = get_datasets(options, dataset=dataset)
config = get_config(trainer=ConstrainedAAE, options=options, optimizer='ADAM', intermediateResolutions=[16, 16], dropout_rate=0.1, dataset=datasetHC)

config.kappa = 1.0
config.scale = 10.0
config.rho = 1.0

# Create an instance of the model and train it
model = ConstrainedAAE(tf.Session(), config, network=constrained_adversarial_autoencoder_Chen)

# Train it
model.train(datasetHC)

# Evaluate
Evaluation.evaluate(datasetPC, model, options, description=f"{type(datasetHC).__name__}-{options['threshold']}", epoch=str(options['train']['numEpochs']))
