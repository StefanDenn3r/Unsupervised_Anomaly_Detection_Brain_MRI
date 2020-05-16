#!/usr/bin/env python
import tensorflow as tf

from models.gaussian_mixture_variational_autoencoder_You import gaussian_mixture_variational_autoencoder_You
from trainers.GMVAE_spatial import GMVAE_spatial
from utils import Evaluation
from utils.default_config_setup import get_config, get_options, get_datasets, Dataset

tf.reset_default_graph()
dataset = Dataset.BRAINWEB
options = get_options(batchsize=8, learningrate=5e-5, numEpochs=1, zDim=128, outputWidth=128, outputHeight=128)
options['data']['dir'] = options["globals"][dataset.value]
datasetHC, datasetPC = get_datasets(options, dataset=dataset)
config = get_config(trainer=GMVAE_spatial, options=options, optimizer='ADAM', intermediateResolutions=[8, 8], dropout_rate=0.1, dataset=datasetHC)

config.dim_c = 9
config.dim_z = 1
config.dim_w = 1
config.c_lambda = 1
config.restore_lr = 1e-3
config.restore_steps = 0
config.tv_lambda = -1.0

# Create an instance of the model and train it
model = GMVAE_spatial(tf.Session(), config, network=gaussian_mixture_variational_autoencoder_You)

# Train it
model.train(datasetHC)

# Evaluate
Evaluation.evaluate(datasetPC, model, options, description=f"{type(datasetHC).__name__}-{options['threshold']}", epoch=str(options['train']['numEpochs']))
