import csv
import math

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score


def xfrange(start, stop, step):
    i = 0
    while start + i * step < stop:
        yield start + i * step
        i += 1


def compute_prc(predictions, labels, filename=None, plottitle="Precision-Recall Curve"):
    precisions, recalls, thresholds = precision_recall_curve(labels.astype(int), predictions)
    auprc = average_precision_score(labels.astype(int), predictions)

    fig = matplotlib.pyplot.figure()
    matplotlib.pyplot.step(recalls, precisions, color='b', alpha=0.2, where='post')
    matplotlib.pyplot.fill_between(recalls, precisions, step='post', alpha=0.2, color='b')
    matplotlib.pyplot.xlabel('Recall')
    matplotlib.pyplot.ylabel('Precision')
    matplotlib.pyplot.ylim([0.0, 1.05])
    matplotlib.pyplot.xlim([0.0, 1.0])
    matplotlib.pyplot.title(f'{plottitle} (area = {auprc:.2f}.)')
    matplotlib.pyplot.show()

    # save a pdf to disk
    if filename:
        fig.savefig(filename)

        with open(filename + ".csv", mode="w") as csv_file:
            fieldnames = ["Precision", "Recall"]
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()
            for i in range(len(precisions)):
                writer.writerow({"Precision": precisions[i], "Recall": recalls[i]})

    return auprc, precisions, recalls, thresholds


def compute_roc(predictions, labels, filename=None, plottitle="ROC Curve"):
    _fpr, _tpr, _ = roc_curve(labels.astype(int), predictions)
    roc_auc = auc(_fpr, _tpr)

    fig = matplotlib.pyplot.figure()
    matplotlib.pyplot.plot(_fpr, _tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    matplotlib.pyplot.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    matplotlib.pyplot.xlim([0.0, 1.0])
    matplotlib.pyplot.ylim([0.0, 1.05])
    matplotlib.pyplot.xlabel('False Positive Rate')
    matplotlib.pyplot.ylabel('True Positive Rate')
    matplotlib.pyplot.title(plottitle)
    matplotlib.pyplot.legend(loc="lower right")
    matplotlib.pyplot.show()

    # save a pdf to disk
    if filename:
        fig.savefig(filename)

    return roc_auc, _fpr, _tpr, _


def dice(P, G):
    psum = np.sum(P.flatten())
    gsum = np.sum(G.flatten())
    pgsum = np.sum(np.multiply(P.flatten(), G.flatten()))
    score = (2 * pgsum) / (psum + gsum)
    return score


def confusion_matrix(P, G):
    tp = np.sum(np.multiply(P.flatten(), G.flatten()))
    fp = np.sum(np.multiply(P.flatten(), np.invert(G.flatten())))
    fn = np.sum(np.multiply(np.invert(P.flatten()), G.flatten()))
    tn = np.sum(np.multiply(np.invert(P.flatten()), np.invert(G.flatten())))
    return tp, fp, tn, fn


def tpr(P, G):
    tp = np.sum(np.multiply(P.flatten(), G.flatten()))
    fn = np.sum(np.multiply(np.invert(P.flatten()), G.flatten()))
    return tp / (tp + fn)


def fpr(P, G):
    tn = np.sum(np.multiply(np.invert(P.flatten()), np.invert(G.flatten())))
    fp = np.sum(np.multiply(P.flatten(), np.invert(G.flatten())))
    return fp / (fp + tn)


def precision(P, G):
    tp = np.sum(np.multiply(P.flatten(), G.flatten()))
    fp = np.sum(np.multiply(P.flatten(), np.invert(G.flatten())))
    return tp / (tp + fp)


def recall(P, G):
    return tpr(P, G)


def vd(P, G):
    tps = np.multiply(P.flatten(), G.flatten())
    return np.sum(np.abs(np.logical_xor(tps, G.flatten()))) / np.sum(G.flatten())


def compute_dice_curve_recursive(predictions, labels, filename=None, plottitle="DICE Curve", granularity=5):
    scores, threshs = compute_dice_score(predictions, labels, granularity)

    best_score, best_threshold = sorted(zip(scores, threshs), reverse=True)[0]

    min_threshs, max_threshs = min(threshs), max(threshs)
    buffer_range = math.fabs(min_threshs - max_threshs) * 0.02
    x_min, x_max = min(threshs) - buffer_range, max(threshs) + buffer_range
    fig = matplotlib.pyplot.figure()
    matplotlib.pyplot.plot(threshs, scores, color='darkorange', lw=2, label='DICE vs Threshold Curve')
    matplotlib.pyplot.xlim([x_min, x_max])
    matplotlib.pyplot.ylim([0.0, 1.05])
    matplotlib.pyplot.xlabel('Thresholds')
    matplotlib.pyplot.ylabel('DICE Score')
    matplotlib.pyplot.title(plottitle)
    matplotlib.pyplot.legend(loc="lower right")
    matplotlib.pyplot.text(x_max - x_max * 0.01, 1, f'Best dice score at {best_threshold:.5f} with {best_score:.4f}', horizontalalignment='right',
                           verticalalignment='top')
    matplotlib.pyplot.show()

    # save a pdf to disk
    if filename:
        fig.savefig(filename)

    bestthresh_idx = np.argmax(scores)
    return scores[bestthresh_idx], threshs[bestthresh_idx]


def compute_dice_score(predictions, labels, granularity):
    def inner_compute_dice_curve_recursive(start, stop, decimal):
        _threshs = []
        _scores = []
        had_recursion = False

        if decimal == granularity:
            return _threshs, _scores

        for i, t in enumerate(xfrange(start, stop, (1.0 / (10.0 ** decimal)))):
            score = dice(np.where(predictions > t, 1, 0), labels)
            if i >= 2 and score <= _scores[i - 1] and not had_recursion:
                _subthreshs, _subscores = inner_compute_dice_curve_recursive(_threshs[i - 2], t, decimal + 1)
                _threshs.extend(_subthreshs)
                _scores.extend(_subscores)
                had_recursion = True
            _scores.append(score)
            _threshs.append(t)

        return _threshs, _scores

    threshs, scores = inner_compute_dice_curve_recursive(0, 1.0, 1)
    sorted_pairs = sorted(zip(threshs, scores))
    threshs, scores = list(zip(*sorted_pairs))
    return scores, threshs


# Predictive pixel-wise variance combining aleatoric and epistemic model uncertainty
# As seen in "What Uncertainties Do we Need in Bayesian Deep Learning for Computer Vision"
# p is a tensor of n monte carlo regression results
# sigma is the same for variances predicted by the network
# axis defines the index of the axis which stores the monte carlo samples
def combined_predictive_uncertainty(p, sigmas, axis=-1, log_var=False):
    if log_var:
        sigmas = np.exp(sigmas)
    return np.mean(np.square(p), axis=axis) - np.square(np.mean(p, axis=axis)) + np.mean(sigmas, axis=axis)
