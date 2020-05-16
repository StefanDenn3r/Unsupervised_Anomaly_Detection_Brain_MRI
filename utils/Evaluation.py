import math
import os
import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
import scipy
import scipy.misc
import scipy.ndimage
import scipy.signal
from imageio import imwrite
from skimage.measure import regionprops, label

from trainers import Metrics
from utils import image_utils, utils


def should(dictionary, key):
    return key in dictionary and dictionary[key]


def get_eval_dictionary():
    _eval = {
        'x': [],
        'reconstructions': [],
        'diffs': [],
        'epistemic_variance': [],
        'labelmaps': [],
        'reconstructionTimes': [],
        'l1reconstructionErrors': [],
        'l1reconstructionErrorMean': 0.0,
        'l1reconstructionErrorSigma': 0.0,
        'l2reconstructionErrors': [],
        'l2reconstructionErrorMean': 0.0,
        'l2reconstructionErrorSigma': 0.0,
        'TP': 0,
        'FP': 0,
        'FN': 0,
        'TN': 0,
        'Dice': [],
        'TPCC': 0,
        'FPCC': 0,
        'FNCC': 0
    }
    return _eval


def merge_eval_dictionaries(eval_dict, _eval_dict):
    for k in eval_dict:
        if isinstance(eval_dict[k], np.ndarray):
            eval_dict[k] = np.concatenate((eval_dict[k], _eval_dict[k]), axis=0)
        elif isinstance(eval_dict[k], list):
            if isinstance(_eval_dict[k], list):
                eval_dict[k] += _eval_dict[k]
            else:
                eval_dict[k] += [_eval_dict[k]]

    return eval_dict


def is_float(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


def squash_intensities(img):
    # logistic function intended to squash reconstruction errors from [0;0.2] to [0;1] (just an example)
    k = 100
    offset = 0.5
    return 2.0 * ((1.0 / (1.0 + np.exp(-k * img))) - offset)


def add_colorbar(img):
    for i in range(img.shape[0]):
        img[i, -1] = float(i) / img.shape[0]

    return img


def apply_brainmask(x, brainmask, erode=True):
    strel = scipy.ndimage.generate_binary_structure(2, 1)
    brainmask = np.expand_dims(brainmask, 2)
    if erode:
        brainmask = scipy.ndimage.morphology.binary_erosion(np.squeeze(brainmask), structure=strel, iterations=12)
    return np.multiply(np.squeeze(brainmask), np.squeeze(x))


def postprocess_slice(x, x_rec, slice_skullmap=None):
    if slice_skullmap is None:
        brainmasks = np.ones(x.shape)
    else:
        strel = scipy.ndimage.generate_binary_structure(2, 1)
        brainmasks = np.expand_dims(slice_skullmap, 3)
        brainmasks = scipy.ndimage.morphology.binary_erosion(np.squeeze(brainmasks), structure=strel, iterations=12)

    x_prior = np.squeeze(x < 0.6)

    x_diff = np.multiply(np.squeeze(brainmasks), np.squeeze(x - x_rec))
    x_diff[x_diff < 0] = 0
    x_diff[x_prior] = 0
    return x_diff


def apply_3d_median_filter(volume, kernelsize=5):  # kernelsize 5 works quite well
    volume = scipy.ndimage.filters.median_filter(volume, (kernelsize, kernelsize, kernelsize))
    return volume


def filter_3d_connected_components(volume):
    sz = None
    if volume.ndim > 3:
        sz = volume.shape
        volume = np.reshape(volume, [sz[0] * sz[1], sz[2], sz[3]])

    cc_volume = label(volume, connectivity=3)
    props = regionprops(cc_volume)
    for prop in props:
        if prop['filled_area'] <= 7:
            volume[cc_volume == prop['label']] = 0

    if sz is not None:
        volume = np.reshape(volume, [sz[0], sz[1], sz[2], sz[3]])
    return volume


def compute_detection_rate(predicted_volume, groundtruth_volume):
    tps = 0
    fns = 0
    fps = 0
    num_slices = groundtruth_volume.shape[0]

    # First, compute intersection of prediction and ground-truth to determine True Positives
    intersected_volume = np.multiply(predicted_volume, groundtruth_volume)

    for s in range(int(math.ceil(num_slices / 20))):

        cc_intersected_volume = label(intersected_volume[s * 20:min((s + 1) * 20, num_slices), :, :])
        props_intersected = regionprops(cc_intersected_volume)

        cc_predicted_volume = label(predicted_volume[s * 20:min((s + 1) * 20, num_slices), :, :])
        props_predicted = regionprops(cc_predicted_volume)

        cc_groundtruth_volume = label(groundtruth_volume[s * 20:min((s + 1) * 20, num_slices), :, :])

        # Filter cc_predicted_volume for any positives which have less than 8 voxels in size
        for pidx, pprop in enumerate(props_predicted):
            if pprop["area"] < 8:
                cc_predicted_volume[cc_predicted_volume == pprop["label"]] = 0

        # Then, remove all the TP connected components from cc_predicted_volume to later determine any False Positives (FPs)
        # Do the same for TP connected components in cc_groundtruth_volume to later be able to determine any False Negatives (FNs)
        for tpidx, tpprop in enumerate(props_intersected):
            coords = tpprop["coords"][0]
            label_in_cc_predicted_volume = cc_predicted_volume[int(coords[0]), int(coords[1]), int(coords[2])]
            cc_predicted_volume[cc_predicted_volume == label_in_cc_predicted_volume] = 0
            label_in_cc_groundtruth_volume = cc_groundtruth_volume[int(coords[0]), int(coords[1]), int(coords[2])]
            cc_groundtruth_volume[cc_groundtruth_volume == label_in_cc_groundtruth_volume] = 0

        # Recompute the regionprops on cc_predicted_volume and cc_groundtruth_volume to determine FPs and FNs
        props_falsely_predicted = regionprops(cc_predicted_volume)
        props_falsely_missed = regionprops(cc_groundtruth_volume)

        # Done
        tps += len(props_intersected)
        fns += len(props_falsely_missed)
        fps += len(props_falsely_predicted)

    return tps, fps, fns


def postprocess_volume(volume):
    volume = scipy.ndimage.filters.median_filter(volume, (5, 5, 5))
    # subvolume = scipy.ndimage.filters.gaussian_filter(subvolume, 3, truncate=3.0)

    # 3D Connected Component Analysis
    return filter_3d_connected_components(volume)


def _evaluate(datasetObj, modelObj, sampleDir, options, split="TEST"):
    os.makedirs(sampleDir, exist_ok=True)

    # Determine the number of testing samples
    num_testing_samples = datasetObj.num_batches(1, set=split)  # batchsize is set to 1 here so we can evaluate per sample
    print("Testing {} samples...".format(num_testing_samples))

    # Setup eval Dictionary
    eval_dict = get_eval_dictionary()

    # Iterate over all patients, and therein, query the desired Nifti files and slices
    patients = [datasetObj.patients[i] for i in datasetObj.get_patient_idx(split=split)]
    for p, patient in enumerate(patients):

        _eval_dict = get_eval_dictionary()

        filtered_files = patient['filtered_files']
        if type(filtered_files) is not list:
            filtered_files = [filtered_files]
        for n, nii_filename in enumerate(filtered_files):
            if len(_eval_dict['diffs']) == 0:
                nii, nii_seg, nii_skullmap = datasetObj.load_volume_and_groundtruth(nii_filename, patient)
                prior_quantile = np.quantile(nii.data, 0.9)

                # Sanity checks - if coregistration went wrong and shapes are bad, we skip this sample
                if min(nii.shape()) < (datasetObj.options.sliceEnd - datasetObj.options.sliceStart):
                    continue

                # Iterate over all slices and collect them
                subvolume = np.zeros(
                    [datasetObj.options.sliceEnd - datasetObj.options.sliceStart, options['train']['outputHeight'],
                     options['train']['outputWidth']])
                subvolume_idx = 0
                slice_start = 0
                slice_end = nii.num_slices_along_axis(datasetObj.options.axis)
                zoom_factor = 1.0
                if datasetObj.options.sliceStart:
                    slice_start = datasetObj.options.sliceStart
                if datasetObj.options.sliceEnd:
                    slice_end = min(datasetObj.options.sliceEnd, nii.num_slices_along_axis(datasetObj.options.axis))
                for s in range(slice_start, slice_end):
                    slice_data = nii.get_slice(s, datasetObj.options.axis)
                    slice_seg = nii_seg.get_slice(s, datasetObj.options.axis).astype(int)
                    slice_skullmap = nii_skullmap.get_slice(s, datasetObj.options.axis).astype(int)

                    if datasetObj.options.sliceResolution is not None:
                        zoom_factor = tuple([i / j for (i, j) in zip(datasetObj.options.sliceResolution, slice_data.shape)])
                        slice_data = scipy.ndimage.zoom(slice_data, zoom_factor)
                        slice_seg = scipy.ndimage.zoom(slice_seg, zoom_factor, mode="nearest")
                        slice_skullmap = scipy.ndimage.zoom(slice_skullmap, zoom_factor, mode="nearest")

                    x = np.expand_dims(slice_data, 2)
                    labelmaps = np.expand_dims(slice_seg, 2)
                    _tmp = time.time()

                    # Monte Carlo Uncertainty Estimation
                    num_samples = 1
                    if should(options, "numMonteCarloSamples"):
                        num_samples = options["numMonteCarloSamples"]
                    x_recs = []
                    x_diffs = []
                    x_log_vars = []
                    results = None
                    for i in range(num_samples):
                        if num_samples > 1:
                            results = modelObj.reconstruct(x, dropout=True)
                        else:
                            results = modelObj.reconstruct(x)
                        x_rec_tmp = results['reconstruction']
                        if "log_var" in results:
                            x_log_vars += [results["log_var"]]

                        x_recs += [np.reshape(apply_brainmask(x_rec_tmp, slice_skullmap, erode=should(options, "erodeBrainmask")),
                                              [1, *datasetObj.options.sliceResolution, 1])]
                        x_diffs += [
                            np.reshape(apply_brainmask(np.maximum(x - x_rec_tmp, 0), slice_skullmap, erode=should(options, "erodeBrainmask")),
                                       [1, *datasetObj.options.sliceResolution, 1])]
                    x_recs = np.array(x_recs)
                    x_diffs = np.array(x_diffs)
                    x_log_vars = np.array(x_log_vars)
                    if x_log_vars.size == 0:
                        x_log_vars = np.zeros(x_diffs.shape)
                    x_recs_var = Metrics.combined_predictive_uncertainty(x_recs, x_log_vars, axis=0, log_var=False)
                    x_recs_var_epistemic = Metrics.combined_predictive_uncertainty(x_recs, np.zeros(x_recs.shape), axis=0, log_var=False)
                    x_recs_mean = np.mean(x_recs, axis=0)

                    x_recs_var = apply_brainmask(x_recs_var, slice_skullmap, erode=should(options, "erodeBrainmask"))

                    x_recs_var_epistemic * (2 * np.expand_dims(np.expand_dims(slice_skullmap, axis=0), axis=-1) - 1)
                    # values outside the brain are getting negative, while values on the brain stay the same

                    _eval_dict['reconstructionTimes'] += [time.time() - _tmp]

                    # Get a sample without dropout
                    x_rec = results['reconstruction']
                    l1err = results['l1err']
                    l2err = results['l2err']
                    if num_samples > 1:
                        x_rec = x_recs_mean
                    if should(options, "keepOnlyPositiveResiduals"):
                        x_diff = np.maximum(x - x_rec, 0)
                    else:
                        x_diff = np.abs(x - x_rec)
                    x_diff = np.reshape(apply_brainmask(x_diff, slice_skullmap, erode=should(options, "erodeBrainmask")),
                                        [1, *datasetObj.options.sliceResolution, 1])
                    if should(options, "applyHyperIntensityPrior"):
                        x_diff[np.reshape(x, [1, *datasetObj.options.sliceResolution, 1]) < prior_quantile] = 0

                    subvolume[subvolume_idx, :, :] = np.squeeze(x_diff)
                    subvolume_idx += 1

                    # Fill eval array
                    _eval_dict['x'] += [x]
                    if num_samples > 1:
                        _eval_dict['epistemic_variance'] += [x_recs_var_epistemic]
                    _eval_dict['reconstructions'] += [x_rec]
                    _eval_dict['labelmaps'] += [np.squeeze(labelmaps)]
                    _eval_dict['l1reconstructionErrors'] += [l1err]
                    _eval_dict['l2reconstructionErrors'] += [l2err]
                    imwrite(os.path.join(sampleDir, '{}_{}.png'.format(p, s)), normalize_and_squeeze(x))
                    imwrite(os.path.join(sampleDir, '{}_{}_rec.png'.format(p, s)), normalize_and_squeeze(x_rec))
                    imwrite(os.path.join(sampleDir, '{}_{}_gt.png'.format(p, s)), normalize_and_squeeze(labelmaps))  # check if normalization is useful
                    imwrite(os.path.join(sampleDir, '{}_{}_diff.png'.format(p, s)), normalize_and_squeeze(x_diff))
                    imwrite(os.path.join(sampleDir, '{}_{}_rec_variance_combined.png'.format(p, s)),
                            np.squeeze(utils.apply_colormap(x_recs_var, plt.cm.jet)))
                    if x_log_vars.size > 0:
                        imwrite(os.path.join(sampleDir, '{}_{}_logvar.png'.format(p, s)), normalize_and_squeeze(np.mean(x_log_vars, axis=0)))

                if should(options, "medianFiltering"):
                    subvolume = apply_3d_median_filter(subvolume)

                _eval_dict['diffs'] += [subvolume]

                for s in range(datasetObj.options.sliceStart, min(datasetObj.options.sliceEnd, nii.num_slices_along_axis(datasetObj.options.axis))):
                    imwrite(os.path.join(sampleDir, '{}_{}_diff_filtered.png'.format(p, s)),
                            normalize_and_squeeze(subvolume[s - datasetObj.options.sliceStart]))
                    squashed = squash_intensities(np.squeeze(subvolume[s - datasetObj.options.sliceStart]))
                    squashed = add_colorbar(squashed)
                    imwrite(os.path.join(sampleDir, '{}_{}_heatmap.png'.format(p, s)), np.squeeze(utils.apply_colormap(squashed, plt.cm.jet)))

                if should(options, "exportVolumes"):
                    dezoom_factor = tuple([1]) + tuple(1 / np.asarray(zoom_factor))
                    subvolume_deprocessed = scipy.ndimage.interpolation.zoom(subvolume, dezoom_factor)
                    nii_seg.set_to_zero()
                    nii_seg.cast_to_float()
                    nii_seg.set_subvolume(datasetObj.options.sliceStart, datasetObj.options.sliceEnd, subvolume_deprocessed,
                                          axis=datasetObj.options.axis)
                    nii_seg.save(os.path.join(sampleDir, '{}.nii.gz'.format(patient['name'])))
                    if options['threshold'] and is_float(options['threshold']):
                        nii_seg.data = np.asarray((nii_seg.data > options['threshold'])).astype(np.float32)
                        nii_seg.update_sitk()
                        nii_seg.save(os.path.join(sampleDir, '{}.binary.nii.gz'.format(patient['name'])))

            # Update the total eval_dict
            eval_dict['x'] += _eval_dict['x']
            eval_dict['diffs'] += _eval_dict['diffs']
            eval_dict['reconstructions'] += _eval_dict['reconstructions']
            eval_dict['labelmaps'] += _eval_dict['labelmaps']
            eval_dict['l1reconstructionErrors'] += _eval_dict['l1reconstructionErrors']
            eval_dict['l2reconstructionErrors'] += _eval_dict['l2reconstructionErrors']
            if "epistemic_variance" in _eval_dict and len(_eval_dict["epistemic_variance"]) > 0:
                eval_dict['epistemic_variance'] += _eval_dict['epistemic_variance']

    print("Done.")

    # Convert list of numpy arrays to numpy array
    eval_dict['x'] = np.squeeze(np.array(eval_dict['x']))
    eval_dict['reconstructions'] = np.squeeze(np.array(eval_dict['reconstructions']))
    eval_dict['diffs'] = np.squeeze(np.array(eval_dict['diffs']))
    if eval_dict['diffs'].ndim > 3:
        eval_dict['diffs'] = np.reshape(eval_dict['diffs'], [eval_dict['diffs'].shape[0] * eval_dict['diffs'].shape[1],
                                                             eval_dict['diffs'].shape[2], eval_dict['diffs'].shape[3]])
    eval_dict['labelmaps'] = np.squeeze(np.array(eval_dict['labelmaps']))
    if "epistemic_variance" in eval_dict and len(eval_dict["epistemic_variance"]) > 0:
        eval_dict['epistemic_variance'] = np.squeeze(np.array(eval_dict['epistemic_variance']))

    # Computer average reconstruction error s etc
    eval_dict['l1reconstructionErrorMean'] = np.mean(eval_dict['l1reconstructionErrors'])
    eval_dict['l1reconstructionErrorVariance'] = np.var(eval_dict['l1reconstructionErrors'])
    eval_dict['l2reconstructionErrorMean'] = np.mean(eval_dict['l2reconstructionErrors'])
    eval_dict['l2reconstructionErrorVariance'] = np.var(eval_dict['l2reconstructionErrors'])
    eval_dict['reconstructionTimes'] = np.mean(np.array(eval_dict['reconstructionTimes']))
    return eval_dict, patients


def normalize_and_squeeze(x):
    return np.squeeze(cv2.normalize(x, None, 0, 255, norm_type=cv2.NORM_MINMAX)).astype('uint8')


def evaluate(datasetPC, gan, options, epoch='last', description=None):
    _time = {'evaluation': time.time()}

    # Variables
    histogram_range = (0.01, 0.075)
    num_slices = options["sliceEnd"] - options["sliceStart"]

    # Create eval folder
    eval_dir = os.path.join(
        options['train']['samplesDir'],
        gan.network.__name__,
        gan.model_dir,
        'eval-' + str(epoch) + '-' + str(utils.timestamp()).replace(":", "-")
    )
    if description is not None:
        eval_dir += "-" + str(description)
    if not os.path.exists(eval_dir):
        os.makedirs(eval_dir)

    # EVALUATE LESION SAMPLES #
    sample_dir = os.path.join(eval_dir, 'samples_test_PC')
    if not os.path.exists(sample_dir):
        os.makedirs(sample_dir)

    eval_pc, patients_pc = _evaluate(datasetPC, gan, sample_dir, options, split="TEST")

    print("Computing histogram for lesion testing difference images")
    eval_pc['diffHistogram'], _ = np.histogram(eval_pc['diffs'], bins='auto', range=histogram_range)
    utils.plot_histogram_with_labels(eval_pc['diffs'], eval_pc['labelmaps'], 'auto', histogram_range,
                                     "Histogram of difference images in the lesion testing dataset",
                                     exportPDF=os.path.join(eval_dir, 'testing_lesions_diffimages_histogram.pdf'))
    print("Done.")
    if "epistemic_variance" in eval_pc and len(eval_pc["epistemic_variance"]) > 0:
        print("Computing uncertainty histogram for lesion testing difference images")
        percentil_99 = np.percentile(eval_pc['epistemic_variance'][eval_pc['epistemic_variance'] >= 0], 99.8)
        _range = (1e-5, percentil_99)
        eval_pc['uncertaintyHistogram'], _ = np.histogram(eval_pc['epistemic_variance'], bins=50, range=_range)
        utils.plot_histogram_with_labels(eval_pc['epistemic_variance'], eval_pc['labelmaps'], 50, _range,
                                         "Histogram of Epistemic Variances images in the lesion testing dataset",
                                         exportPDF=os.path.join(eval_dir, 'testing_lesions_epistemic_variances_histogram.pdf'))
        print("Done.")

    print("Computing ROC curve for Lesion samples")
    _time['ROC'] = time.time()
    eval_pc['diff_AUC'], _fpr, _tpr, _threshs = Metrics.compute_roc(eval_pc['diffs'].flatten(), eval_pc['labelmaps'].astype(bool).flatten(),
                                                                    plottitle="ROC Curve for Lesion Testing Samples",
                                                                    filename=os.path.join(eval_dir, 'rocPC.png'))
    _time['ROC'] = time.time() - _time['ROC']
    print('Done in {} seconds'.format(_time['ROC']))
    if should(options, "exportROC"):
        _tmp = {"fpr": _fpr, "tpr": _tpr, "threshs": _threshs}
        np.save(os.path.join(eval_dir, 'rocPC.npy'), _tmp, allow_pickle=True)

    print("Computing Precision-Recall curve for Lesion samples")
    _time['PRC'] = time.time()
    eval_pc['diff_AUPRC'], _precisions, _recalls, _threshs = Metrics.compute_prc(
        eval_pc['diffs'].flatten(),
        eval_pc['labelmaps'].astype(bool).flatten(),
        plottitle="Precision-Recall Curve for Lesion Testing Samples",
        filename=os.path.join(eval_dir, 'prcPC.png')
    )
    _time['PRC'] = time.time() - _time['PRC']
    print('Done in {} seconds'.format(_time['PRC']))
    if should(options, "exportPRC"):
        _tmp = {"precisions": _precisions, "recalls": _recalls, "threshs": _threshs}
        np.save(os.path.join(eval_dir, 'prcPC.npy'), _tmp, allow_pickle=True)
    # Quickly determine thresholds for different precisions to get the maximal possible recall
    idx_precision70 = np.argmax(_precisions <= 0.7)
    diffs_thresholded_at_precision70 = filter_3d_connected_components(np.squeeze(eval_pc['diffs'] > _threshs[idx_precision70]))

    print("Computing DICE curve for Lesion samples")
    _time['DiceCurve'] = time.time()
    eval_pc['bestDiceScore'], eval_pc['bestThreshold'] = Metrics.compute_dice_curve_recursive(
        eval_pc['diffs'].flatten(), eval_pc['labelmaps'].flatten(),
        plottitle="DICE vs Thresholds Curve for Lesion Testing Samples",
        filename=os.path.join(eval_dir, 'dicePC.png'),
        granularity=10
    )
    _time['DiceCurve'] = time.time() - _time['DiceCurve']
    print('Done in {} seconds'.format(_time['DiceCurve']))

    if options["threshold"] == 'bestdice':
        diffs_thresholded = eval_pc['diffs'] > eval_pc['bestThreshold']
    else:
        diffs_thresholded = eval_pc['diffs'] > options["threshold"]
        diffs_thresholded_at_precision70 = diffs_thresholded
    diffs_thresholded = filter_3d_connected_components(np.squeeze(diffs_thresholded))

    eval_pc['thresholdType'] = options["threshold"]
    eval_pc['DiceScore'] = Metrics.dice(diffs_thresholded, eval_pc['labelmaps'])
    eval_pc['DiceScorePerPatient'] = []
    eval_pc['PrecisionPerPatient'] = []
    eval_pc['RecallPerPatient'] = []
    for p, patient in enumerate(patients_pc):
        subvolume_prediction = diffs_thresholded[p * num_slices:(p + 1) * num_slices, :, :]
        subvolume_groundtruth = eval_pc['labelmaps'][p * num_slices:(p + 1) * num_slices, :, :]
        eval_pc['DiceScorePerPatient'] += [Metrics.dice(subvolume_prediction, subvolume_groundtruth.astype(bool))]
        eval_pc['PrecisionPerPatient'] += [Metrics.precision(subvolume_prediction, subvolume_groundtruth.astype(bool))]
        eval_pc['RecallPerPatient'] += [Metrics.recall(subvolume_prediction, subvolume_groundtruth.astype(bool))]

        # Choose a different operating point from the Precision Recall Curve!
        # e.g. determine the threshold at 20% Precision and base don that, this lesion detection rate
        _TPs, _FPs, _FNs = compute_detection_rate(np.squeeze(diffs_thresholded_at_precision70[p * num_slices:(p + 1) * num_slices, :, :]),
                                                  np.squeeze(subvolume_groundtruth.astype(bool)))
        eval_pc['TPCC'] += _TPs
        eval_pc['FPCC'] += _FPs
        eval_pc['FNCC'] += _FNs
    eval_pc['DiceScorePerPatientMean'] = np.mean(np.array(eval_pc['DiceScorePerPatient']))
    eval_pc['DiceScorePerPatientStd'] = np.std(np.array(eval_pc['DiceScorePerPatient']))
    eval_pc['PrecisionPerPatientMean'] = np.mean(np.array(eval_pc['PrecisionPerPatient']))
    eval_pc['PrecisionPerPatientStd'] = np.std(np.array(eval_pc['PrecisionPerPatient']))
    eval_pc['RecallPerPatientMean'] = np.mean(np.array(eval_pc['RecallPerPatient']))
    eval_pc['RecallPerPatientStd'] = np.std(np.array(eval_pc['RecallPerPatient']))

    # Threshold diffs and compute Confusion matrix, TPR, FPR and VolumeDifference
    eval_pc['TP'], eval_pc['FP'], eval_pc['TN'], eval_pc['FN'] = Metrics.confusion_matrix(
        diffs_thresholded, eval_pc['labelmaps'].astype(bool))
    eval_pc['TPR'] = Metrics.tpr(diffs_thresholded, eval_pc['labelmaps'].astype(bool))
    eval_pc['FPR'] = Metrics.tpr(diffs_thresholded, eval_pc['labelmaps'].astype(bool))
    eval_pc['VD'] = Metrics.vd(diffs_thresholded, eval_pc['labelmaps'].astype(bool))
    if eval_pc['TPCC'] + eval_pc['FNCC'] > 0:
        eval_pc['TPRCC'] = eval_pc['TPCC'] / (eval_pc['TPCC'] + eval_pc['FNCC'])
    else:
        eval_pc['TPRCC'] = 0.0
    if eval_pc['TPCC'] + eval_pc['FPCC'] > 0:
        eval_pc['PrecisionCC'] = eval_pc['TPCC'] / (eval_pc['TPCC'] + eval_pc['FPCC'])
    else:
        eval_pc['PrecisionCC'] = 0.0

    for idx in range(0, eval_pc['x'].shape[0]):
        tmp = image_utils.augment_prediction_and_groundtruth_to_image(eval_pc['x'][idx],
                                                                      diffs_thresholded[idx],
                                                                      eval_pc['labelmaps'][idx])
        p = math.floor(float(idx) / num_slices)
        s = datasetPC.options.sliceStart + (idx % (datasetPC.options.sliceEnd - datasetPC.options.sliceStart))
        imwrite(os.path.join(sample_dir, '{}_{}_vis.png'.format(p, s)), np.squeeze(cv2.normalize(tmp, None, 0, 255)).astype('uint8'))

    # Store evalPC to disk

    eval_pc.pop('x')
    eval_pc.pop('diffs')
    eval_pc.pop('labelmaps')
    eval_pc.pop('l1reconstructionErrors')
    eval_pc.pop('l2reconstructionErrors')
    eval_pc.pop('reconstructions')
    eval_pc.pop('diffHistogram')

    np.save(os.path.join(eval_dir, 'evalPC.npy'), eval_pc)

    _time['evaluation'] = time.time() - _time['evaluation']

    # Export to TXT
    f = open(os.path.join(eval_dir, 'evalPC.txt'), "w")
    f.write(str(eval_pc))
    f.close()


def determine_threshold_on_labeled_patients(dataset_pc, model, options, epoch='last', description=None):
    # Create eval folder
    eval_dir = os.path.join(
        options['train']['samplesDir'],
        model.network.__name__,
        model.model_dir,
        'eval-' + str(epoch) + '-' + str(utils.timestamp()).replace(":", "-")
    )
    if description is not None:
        eval_dir += "-" + str(description)
    if not os.path.exists(eval_dir):
        os.makedirs(eval_dir)

    sample_dir = os.path.join(eval_dir, 'samples_val_PC')
    if not os.path.exists(sample_dir):
        os.makedirs(sample_dir)

    if not isinstance(dataset_pc, list):
        dataset_pc = [dataset_pc]

    eval_pc_val = None
    patients_pc_val = None
    for i, ds in enumerate(dataset_pc):
        if i == 0:
            eval_pc_val, patients_pc_val = _evaluate(ds, model, sample_dir, options, split="VAL")
        else:
            _eval_pc_val, _patients_pc_val = _evaluate(ds, model, sample_dir, options, split="VAL")
            eval_pc_val = merge_eval_dictionaries(eval_pc_val, _eval_pc_val)
            patients_pc_val += [_patients_pc_val]

    print("Computing DICE curve for Lesion Validation samples")
    eval_pc_val['bestDiceScore'], eval_pc_val['bestThreshold'] = Metrics.compute_dice_curve_recursive(
        eval_pc_val['diffs'].flatten(),
        eval_pc_val['labelmaps'].flatten(),
        plottitle="DICE vs Thresholds Curve for Lesion Testing Validation Samples",
        filename=os.path.join(eval_dir, 'dicePC_VAL.png'),
        granularity=10
    )
    return eval_pc_val['bestDiceScore'], eval_pc_val['bestThreshold']
