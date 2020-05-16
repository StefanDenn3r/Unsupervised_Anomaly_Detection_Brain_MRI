import calendar
import csv
import datetime
import pickle
import time

import cv2
import matplotlib.pyplot
import numpy as np
from PIL import Image
from matplotlib.backends.backend_pdf import PdfPages

COLORS = ['b', 'r', 'g', 'c']


def timestamp():
    ts = calendar.timegm(time.gmtime())
    return datetime.datetime.fromtimestamp(ts).isoformat()


def apply_colormap(img, colormap_handle):
    img = img - img.min()
    if img.max() != 0:
        img = img / img.max()
    img = Image.fromarray(np.uint8(colormap_handle(img) * 255))
    return img


def plot_histogram(data, bins, range, title, exportPDF=None):
    f = matplotlib.pyplot.figure()
    matplotlib.pyplot.hist(data.flatten(), bins=bins, range=range, density=True)  # arguments are passed to np.histogram
    matplotlib.pyplot.title(title)
    matplotlib.pyplot.show(block=False)

    # save a pdf to disk
    if exportPDF:
        pp = PdfPages(exportPDF)
        pp.savefig(f)
        pp.close()

    return f


def plot_histogram_with_labels(data, labels, bins, _range, title, exportPDF=None):
    classes = np.unique(labels)
    f = matplotlib.pyplot.figure()

    for i in range(classes.size):
        data_with_current_label = data[labels == classes[i]]
        n, bins, patches = matplotlib.pyplot.hist(data_with_current_label.flatten(), bins=bins, range=_range, color=COLORS[i])

        with open(f'{exportPDF.split(".")[0]}.{i}.npy', 'wb') as file:
            pickle.dump({'n': n, 'bins': bins, 'mean': np.mean(data_with_current_label), 'var': np.var(data_with_current_label)}, file)

        with open(exportPDF + ".{}.csv".format(i), mode="w") as csv_file:
            fieldnames = ["Bin", "Count"]
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()
            for k in range(len(n)):
                writer.writerow({"Bin": bins[k], "Count": n[k]})

    matplotlib.pyplot.title(title)
    matplotlib.pyplot.show(block=False)

    # save a pdf to disk
    if exportPDF:
        pp = PdfPages(exportPDF)
        pp.savefig(f)
        pp.close()

    return f


def normalize(x):
    return np.expand_dims(cv2.normalize(x, None, 0, 1, cv2.NORM_MINMAX), -1)
