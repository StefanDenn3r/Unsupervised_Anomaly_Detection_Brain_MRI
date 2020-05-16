#!/usr/bin/env python
import tensorflow as tf

from models import context_encoder_variational_autoencoder_Zimmerer
from trainers.ceVAE import ceVAE
from utils import Evaluation
from utils.default_config_setup import get_config, get_options, get_datasets, Dataset

tf.reset_default_graph()
dataset = Dataset.BRAINWEB
options = get_options(batchsize=8, learningrate=0.0001, numEpochs=1, zDim=128, outputWidth=128, outputHeight=128)
options['data']['dir'] = options["globals"][dataset.value]
datasetHC, datasetPC = get_datasets(options, dataset=dataset)
config = get_config(trainer=ceVAE, options=options, optimizer='ADAM', intermediateResolutions=[8, 8], dropout_rate=0.1, dataset=datasetHC)

# Create an instance of the model and train it
model = ceVAE(tf.Session(), config, network=context_encoder_variational_autoencoder_Zimmerer.context_encoder_variational_autoencoder_Zimmerer)

# Train it
model.train(datasetHC)

# Evaluate
Evaluation.evaluate(datasetPC, model, options, description=f"{type(datasetHC).__name__}-{options['threshold']}", epoch=str(options['train']['numEpochs']))
