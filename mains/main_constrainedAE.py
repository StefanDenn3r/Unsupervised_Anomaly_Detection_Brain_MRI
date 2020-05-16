#!/usr/bin/env python
import tensorflow as tf

from models.constrained_autoencoder import constrained_autoencoder
from trainers.ConstrainedAE import ConstrainedAE
from utils import Evaluation
from utils.default_config_setup import get_config, get_options, get_datasets, Dataset

# reset default graph
tf.reset_default_graph()

tf.reset_default_graph()
dataset = Dataset.BRAINWEB
options = get_options(batchsize=8, learningrate=0.001, numEpochs=1, zDim=1024, outputWidth=128, outputHeight=128)
options['data']['dir'] = options["globals"][dataset.value]
datasetHC, datasetPC = get_datasets(options, dataset=dataset)
config = get_config(trainer=ConstrainedAE, options=options, optimizer='ADAM', intermediateResolutions=[16, 16], dropout_rate=0.1, dataset=datasetHC)

config.rho = 1

# Create an instance of the model and train it
model = ConstrainedAE(tf.Session(), config, network=constrained_autoencoder)

# Train it
model.train(datasetHC)

# Evaluate
Evaluation.evaluate(datasetPC, model, options, description=f"{type(datasetHC).__name__}-{options['threshold']}", epoch=str(options['train']['numEpochs']))
