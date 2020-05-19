#!/usr/bin/env python

import argparse
import json
import os
import sys
from importlib.machinery import SourceFileLoader
from typing import Tuple

import tensorflow as tf

from utils.Evaluation import evaluate, determine_threshold_on_labeled_patients
from utils.default_config_setup import get_config, get_options, get_datasets, Dataset

base_path = os.path.dirname(os.path.abspath(__file__))


def main(args):
    # reset default graph
    tf.reset_default_graph()
    base_path_trainer = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'trainers', f'{args.trainer}.py')
    base_path_network = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models', f'{args.model}.py')
    trainer = getattr(SourceFileLoader(args.trainer, base_path_trainer).load_module(), args.trainer)
    network = getattr(SourceFileLoader(args.model, base_path_network).load_module(), args.model)

    with open(os.path.join(base_path, args.config), 'r') as f:
        json_config = json.load(f)

    dataset = Dataset.BRAINWEB
    options = get_options(batchsize=args.batchsize, learningrate=args.lr, numEpochs=args.numEpochs, zDim=args.zDim, outputWidth=args.outputWidth,
                          outputHeight=args.outputHeight, slices_start=args.slices_start, slices_end=args.slices_end,
                          numMonteCarloSamples=args.numMonteCarloSamples, config=json_config)
    options['data']['dir'] = options["globals"][dataset.value]
    dataset_hc, dataset_pc = get_datasets(options, dataset=dataset)
    config = get_config(
        trainer=trainer,
        options=options,
        optimizer=args.optimizer,
        intermediateResolutions=args.intermediateResolutions,
        dropout_rate=0.2,
        dataset=dataset_hc
    )

    # handle additional Config parameters e.g. for GMVAE
    for arg in vars(args):
        if hasattr(config, arg):
            setattr(config, arg, getattr(args, arg))

    # Create an instance of the model and train it
    model = trainer(tf.Session(), config, network=network)

    # Train it
    model.train(dataset_hc)

    ########################
    #  Evaluate best dice  #
    #########################
    if not args.threshold:
        # if no threshold is given but a dataset => Best dice evaluation on specific dataset
        if args.ds:
            # evaluate specific dataset
            evaluate_optimal(model, options, args.ds)
            return
        else:
            # evaluate all datasets for best dice without hyper intensity prior
            options['applyHyperIntensityPrior'] = False
            evaluate_optimal(model, options, Dataset.Brainweb)
            evaluate_optimal(model, options, Dataset.MSLUB)
            evaluate_optimal(model, options, Dataset.MSISBI2015)

            # evaluate all datasets for best dice without hyper intensity prior
            options['applyHyperIntensityPrior'] = True
            evaluate_optimal(model, options, Dataset.Brainweb)
            evaluate_optimal(model, options, Dataset.MSLUB)
            evaluate_optimal(model, options, Dataset.MSISBI2015)

    ###############################################
    #  Evaluate generalization to other datasets  #
    ###############################################
    if args.threshold and args.ds:  # only threshold is invalid
        evaluate_with_threshold(model, options, args.threshold, args.ds)
    else:
        options['applyHyperIntensityPrior'] = False
        datasetBrainweb = get_evaluation_dataset(options, Dataset.Brainweb)
        _bestDiceVAL, _threshVAL = determine_threshold_on_labeled_patients([datasetBrainweb], model, options, description='VAL')

        print(f"Optimal threshold on MS Lesion Validation Set without optimal postprocessing: {_threshVAL} (Dice-Score {_bestDiceVAL})")

        # Re-evaluate with the previously determined threshold
        evaluate_with_threshold(model, options, _threshVAL, Dataset.Brainweb)
        evaluate_with_threshold(model, options, _threshVAL, Dataset.MSLUB)
        evaluate_with_threshold(model, options, _threshVAL, Dataset.MSISBI2015)


def evaluate_with_threshold(model, options, threshold, dataset):
    options['applyHyperIntensityPrior'] = False
    options['threshold'] = threshold
    description = lambda ds: f'{type(ds).__name__}-VALthresh_{options["threshold"]}'
    evaluation_dataset = get_evaluation_dataset(options, dataset)
    evaluate(evaluation_dataset, model, options, description=description(evaluation_dataset), epoch=str(options['train']['numEpochs']))


def evaluate_optimal(model, options, dataset):
    hyper_intensity_prior_str = ''
    if options['applyHyperIntensityPrior']:
        hyper_intensity_prior_str = "_wPrior"
    evaluation_dataset = get_evaluation_dataset(options, dataset)
    epochs = str(options['train']['numEpochs'])
    description = f'{type(evaluation_dataset).__name__}_upperbound_{options["threshold"]}{hyper_intensity_prior_str}'
    # Evaluate
    evaluate(evaluation_dataset, model, options, description=description, epoch=epochs)


def get_evaluation_dataset(options, dataset=Dataset.BRAINWEB):
    options['data']['dir'] = options["globals"][dataset.value]
    return get_datasets(options, dataset=dataset)[1]


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='Framework')
    args.print_help(sys.stderr)
    args.add_argument('-c', '--config', default='config.default.json', type=str, help='config-path')
    args.add_argument('-b', '--batchsize', default=8, type=int, help='batchsize')
    args.add_argument('-l', '--lr', default=0.0001, type=float, help='learning rate')
    args.add_argument('-E', '--numEpochs', default=1000, type=int, help='how many epochs to train')
    args.add_argument('-z', '--zDim', default=128, type=int, help='Latent dimension')
    args.add_argument('-w', '--outputWidth', default=128, type=int, help='Output width')
    args.add_argument('-g', '--outputHeight', default=128, type=int, help='Output height')
    args.add_argument('-o', '--optimizer', default='ADAM', type=str, help='Can be either ADAM, SGD or RMSProp')
    args.add_argument('-i', '--intermediateResolutions', default=(8, 8), type=Tuple[int], help='Spatial Bottleneck resolution')
    args.add_argument('-s', '--slices_start', default=20, type=int, help='slices start')
    args.add_argument('-e', '--slices_end', default=130, type=int, help='slices end')
    args.add_argument('-t', '--trainer', default='AE', type=str, help='Can be every class from trainers directory')
    args.add_argument('-m', '--model', default='autoencoder', type=str, help='Can be every class from models directory')
    args.add_argument('-O', '--threshold', default=None, type=float, help='Use predefined ThreshOld')
    args.add_argument('-d', '--ds', default=None, type=Dataset, help='Only evaluate on given dataset')

    # following arguments are only relevant for specific architectures
    args.add_argument('-n', '--numMonteCarloSamples', default=0, type=int, help='Amount of Monte Carlos Samples during restoration')
    args.add_argument('-G', '--use_gradient_based_restoration', default=False, type=bool, help='only for ceVAE')
    args.add_argument('-K', '--kappa', default=1.0, type=float, help='only for GANs')
    args.add_argument('-M', '--scale', default=10.0, type=float, help='only for GANs')
    args.add_argument('-R', '--rho', default=1.0, type=float, help='only for ConstrainedAAE')
    args.add_argument('-C', '--dim_c', default=9, type=int, help='only for GMVAE')
    args.add_argument('-Z', '--dim_z', default=128, type=int, help='only for GMVAE')
    args.add_argument('-W', '--dim_w', default=1, type=int, help='only for GMVAE')
    args.add_argument('-A', '--c_lambda', default=1, type=int, help='only for GMVAE')
    args.add_argument('-L', '--restore_lr', default=1e-3, type=float, help='only for GMVAE')
    args.add_argument('-S', '--restore_steps', default=150, type=int, help='only for GMVAE')
    args.add_argument('-T', '--tv_lambda', default=-1.0, type=float, help='only for GMVAE')

    main(args.parse_args())
