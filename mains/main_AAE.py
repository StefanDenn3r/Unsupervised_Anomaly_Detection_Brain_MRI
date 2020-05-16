#!/usr/bin/env python
import tensorflow as tf

from models.adversarial_autoencoder import adversarial_autoencoder
from trainers.AAE import AAE
from utils import Evaluation
from utils.default_config_setup import get_config, get_options, get_datasets, Dataset

tf.reset_default_graph()
dataset = Dataset.BRAINWEB
options = get_options(batchsize=128, learningrate=0.0001, numEpochs=1, zDim=128, outputWidth=128, outputHeight=128)
options['data']['dir'] = options["globals"][dataset.value]
datasetHC, datasetPC = get_datasets(options, dataset=dataset)
config = get_config(trainer=AAE, options=options, optimizer='ADAM', intermediateResolutions=[16, 16], dropout_rate=0.1, dataset=datasetHC)

config.scale = 10.0

# Create an instance of the model and train it
model = AAE(tf.Session(), config, network=adversarial_autoencoder)

# Train it
model.train(datasetHC)

# Evaluate
Evaluation.evaluate(datasetPC, model, options, description=f"{type(datasetHC).__name__}-{options['threshold']}", epoch=str(options['train']['numEpochs']))
