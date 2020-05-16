import json
import os
from enum import Enum

from dataloaders.BRAINWEB import BRAINWEB
from dataloaders.MSISBI2015 import MSISBI2015
from dataloaders.MSLUB import MSLUB
from dataloaders.MSSEG2008 import MSSEG2008

base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class Dataset(Enum):
    BRAINWEB = 'BRAINWEBDIR'
    MSSEG2008_UNC = 'MSSEG2008DIR'
    MSSEG2008_CHB = 'MSSEG2008DIR'
    MSISBI2015 = 'MSISBI2015DIR'
    MSLUB = 'MSLUBDIR'


def get_options(batchsize, learningrate, numEpochs, zDim, outputWidth, outputHeight, slices_start=20, slices_end=130, numMonteCarloSamples=0, config=None):
    options = {}
    # Load config.json, which should hold DATADIR, CHECKPOINTDIR and SAMPLEDIR
    if config:
        options["globals"] = config
    else:
        with open(os.path.join(base_path, "config.default.json"), 'r') as f:
            options["globals"] = json.load(f)

    # Options
    options['debug'] = False
    options['data'] = {}
    options['train'] = {}
    options['train']['checkpointDir'] = options["globals"]["CHECKPOINTDIR"]
    options['train']['samplesDir'] = options["globals"]["SAMPLEDIR"]
    options['train']['batchsize'] = batchsize
    options['train']['learningrate'] = learningrate
    options['train']['numEpochs'] = numEpochs
    options['train']['zDim'] = zDim
    options['train']['snapshotAfter'] = 1000  # Take a snapshot after every 50 iterations
    options['train']['outputWidth'] = outputWidth
    options['train']['outputHeight'] = outputHeight
    options['train']['useTensorboard'] = True
    options['train']['useMatplotlib'] = False
    options['train']['tensorboardPort'] = 9001
    options['sliceStart'] = slices_start  # 20
    options['sliceEnd'] = slices_end  # 130
    options['threshold'] = 'bestdice'
    options['exportVolumes'] = False
    options['exportPRC'] = True
    options['exportROC'] = True
    options['numMonteCarloSamples'] = numMonteCarloSamples
    options['keepOnlyPositiveResiduals'] = True
    options['applyHyperIntensityPrior'] = True
    options['medianFiltering'] = True
    options['erodeBrainmask'] = True
    return options


def get_datasets(options, dataset: Dataset = Dataset.BRAINWEB):
    if dataset == Dataset.BRAINWEB:
        return get_Brainweb_healthy_dataset(options), get_Brainweb_lesion_dataset(options)
    elif dataset == Dataset.MSSEG2008_UNC:
        return None, get_MSSEG2008_dataset(options, 'UNC')
    elif dataset == Dataset.MSSEG2008_CHB:
        return None, get_MSSEG2008_dataset(options, 'CHB')
    elif dataset == Dataset.MSISBI2015:
        return None, get_MSISBI2015_dataset(options)
    elif dataset == Dataset.MSLUB:
        return None, get_MSLUB_dataset(options)
    else:
        raise ValueError(f'No valid dataset given: {dataset}')


###########################
#       MSSEG2008         #
###########################
def get_MSSEG2008_dataset(options, filter_sanner):
    dataset_options = get_MSSEG2008_dataset_options(options, filter_sanner)
    dataset = MSSEG2008(dataset_options)
    if options['debug']:
        dataset.visualize()

    return dataset


def get_MSSEG2008_dataset_options(options, filter_sanner):
    dataset_options = MSSEG2008.Options()
    dataset_options.description = ''
    dataset_options.debug = options['debug']
    dataset_options.dir = options['globals']['MSSEG2008DIR']
    dataset_options.useCrops = False
    dataset_options.cropType = 'center'  # Crop patches around lesions
    dataset_options.cropWidth = options['train']['outputWidth']
    dataset_options.cropHeight = options['train']['outputHeight']
    dataset_options.numRandomCropsPerSlice = 5  # Not needed when doing center crops
    dataset_options.rotations = [0]
    dataset_options.partition = {'TRAIN': 0, 'VAL': 2, 'TEST': 8}
    dataset_options.sliceResolution = [options['train']['outputHeight'], options['train']['outputWidth']]
    dataset_options.cache = True
    dataset_options.numSamples = -1
    dataset_options.addInstanceNoise = False
    dataset_options.axis = 'axial'
    dataset_options.filterScanner = filter_sanner  # 'UNC'or 'CHB'
    dataset_options.filterProtocols = ['FLAIR']
    dataset_options.filterType = "train"
    dataset_options.normalizationMethod = 'scaling'
    dataset_options.skullStripping = True
    dataset_options.sliceStart = options['sliceStart']
    dataset_options.sliceEnd = options['sliceEnd']
    dataset_options.skullStripping = True
    dataset_options.format = "aligned"

    return dataset_options


###########################
#       MSISBI2015        #
###########################
def get_MSISBI2015_dataset(options):
    dataset_options = get_MSISBI2015_dataset_options(options)
    dataset = MSISBI2015(dataset_options)
    if options['debug']:
        dataset.visualize()

    return dataset


def get_MSISBI2015_dataset_options(options):
    dataset_options = MSISBI2015.Options()
    dataset_options.description = ''
    dataset_options.debug = options['debug']
    dataset_options.dir = options['globals']['MSISBI2015DIR']
    dataset_options.useCrops = False
    dataset_options.cropType = 'center'  # Crop patches around lesions
    dataset_options.cropWidth = options['train']['outputWidth']
    dataset_options.cropHeight = options['train']['outputHeight']
    dataset_options.numRandomCropsPerSlice = 5  # Not needed when doing center crops
    dataset_options.rotations = [0]
    dataset_options.partition = {'TRAIN': 0, 'VAL': 5, 'TEST': 15}
    dataset_options.sliceResolution = [options['train']['outputHeight'], options['train']['outputWidth']]
    dataset_options.cache = True
    dataset_options.numSamples = -1
    dataset_options.addInstanceNoise = False
    dataset_options.axis = 'axial'
    dataset_options.filterProtocols = ['FLAIR']
    dataset_options.filterType = "train"
    dataset_options.normalizationMethod = 'scaling'
    dataset_options.skullStripping = True
    dataset_options.sliceStart = options['sliceStart']
    dataset_options.sliceEnd = options['sliceEnd']
    dataset_options.skullStripping = True
    dataset_options.format = "aligned"
    return dataset_options


###########################
#       MSLUB        #
###########################
def get_MSLUB_dataset(options):
    dataset_options = get_MSLUB_dataset_options(options)
    dataset = MSLUB(dataset_options)
    if options['debug']:
        dataset.visualize()

    return dataset


def get_MSLUB_dataset_options(options):
    dataset_options = MSLUB.Options()
    dataset_options.description = ''
    dataset_options.debug = options['debug']
    dataset_options.dir = options['globals']['MSLUBDIR']
    dataset_options.useCrops = False
    dataset_options.cropType = 'center'  # Crop patches around lesions
    dataset_options.cropWidth = options['train']['outputWidth']
    dataset_options.cropHeight = options['train']['outputHeight']
    dataset_options.numRandomCropsPerSlice = 5  # Not needed when doing center crops
    dataset_options.rotations = [0]
    dataset_options.partition = {'TRAIN': 0, 'VAL': 5, 'TEST': 25}
    dataset_options.sliceResolution = [options['train']['outputHeight'], options['train']['outputWidth']]
    dataset_options.cache = True
    dataset_options.numSamples = -1
    dataset_options.addInstanceNoise = False
    dataset_options.axis = 'axial'
    dataset_options.filterProtocols = ['FLAIR']
    dataset_options.normalizationMethod = 'scaling'
    dataset_options.skullStripping = True
    dataset_options.sliceStart = options['sliceStart']
    dataset_options.sliceEnd = options['sliceEnd']
    dataset_options.skullStripping = True
    dataset_options.format = "aligned"

    return dataset_options


#######################
#      Brainweb       #
#######################
def get_Brainweb_healthy_dataset(options):
    dataset_options = get_Brainweb_dataset_options(options)
    dataset_hc = BRAINWEB(dataset_options)
    if options['debug']:
        dataset_hc.visualize()
    return dataset_hc


def get_Brainweb_lesion_dataset(options):
    dataset_options = get_Brainweb_dataset_options(options)
    # Center Crops of slices from patients with lesions. Only for testing
    dataset_options.partition = {'TRAIN': 0.0, 'VAL': 0.0, 'TEST': 1.0}
    dataset_options.filterType = 'SEVEREMS'
    dataset_options.rotations = [0]
    return BRAINWEB(dataset_options)


def get_Brainweb_dataset_options(options):
    dataset_options = BRAINWEB.Options()
    dataset_options.description = ""
    dataset_options.debug = options['debug']
    dataset_options.dir = options['data']['dir']
    dataset_options.useCrops = False
    dataset_options.cropType = 'center'  # Not used when useCrops is False
    dataset_options.cropWidth = options['train']['outputWidth']
    dataset_options.cropHeight = options['train']['outputHeight']
    dataset_options.numRandomCropsPerSlice = 5  # Not needed when doing center crops
    dataset_options.rotations = [0]
    dataset_options.partition = {'TRAIN': 0.7, 'VAL': 0.3, 'TEST': 0.0}
    dataset_options.sliceResolution = [options['train']['outputHeight'], options['train']['outputWidth']]
    dataset_options.cache = True
    dataset_options.numSamples = -1
    dataset_options.addInstanceNoise = False
    dataset_options.axis = 'axial'
    dataset_options.filterType = 'NORMAL'
    dataset_options.filterProtocol = 'T2'
    dataset_options.normalizationMethod = 'scaling'
    dataset_options.skullRemoval = True
    dataset_options.sliceStart = options['sliceStart']
    dataset_options.sliceEnd = options['sliceEnd']
    dataset_options.backgroundRemoval = True
    dataset_options.registerTo = None
    return dataset_options


def get_config(trainer, options, optimizer, intermediateResolutions, dropout_rate, dataset):
    config = trainer.Config()
    config.dataset = type(dataset).__name__
    config.description = ''
    config.numChannels = dataset.num_channels
    config.batchsize = options['train']['batchsize']
    config.checkpointDir = options['train']['checkpointDir']
    config.snapShotAfter = options['train']['snapshotAfter']
    config.sampleDir = options['train']['samplesDir']
    config.learningrate = options['train']['learningrate']
    config.numEpochs = options['train']['numEpochs']
    config.zDim = options['train']['zDim']
    config.beta1 = 0.5
    config.outputHeight = options['train']['outputHeight']
    config.outputWidth = options['train']['outputWidth']
    config.useTensorboard = options['train']['useTensorboard']
    config.useMatplotlib = options['train']['useMatplotlib']
    config.tensorboardPort = options['train']['tensorboardPort']
    config.debugGradients = options['debug']
    config.optimizer = optimizer
    config.intermediateResolutions = intermediateResolutions
    config.weightRegularization = 0.0
    config.dropout_rate = dropout_rate
    config.dropout = False
    config.l1_weight = 1.0
    config.options = options
    return config
