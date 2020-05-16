import matplotlib.pyplot as plt
import nrrd
import numpy as np


# Class for working with a NII file in the context of machine learning
class NRRD:
    VIEW_MAPPING = {'saggital': 0, 'coronal': 1, 'axial': 2}

    def __init__(self, filename):
        self.data, self.info = nrrd.read(filename)

    @property
    def num_saggital_slices(self):
        return self.data.shape[NRRD.VIEW_MAPPING['saggital']]

    @property
    def num_coronal_slices(self):
        return self.data.shape[NRRD.VIEW_MAPPING['coronal']]

    @property
    def num_axial_slices(self):
        return self.data.shape[NRRD.VIEW_MAPPING['axial']]

    @staticmethod
    def set_view_mapping(mapping):
        NRRD.VIEW_MAPPING = mapping

    def shape(self):
        return self.data.shape

    @staticmethod
    def get_axis_index(axis):
        return NRRD.VIEW_MAPPING[axis]

    def num_slices_along_axis(self, axis):
        return self.data.shape[NRRD.VIEW_MAPPING[axis]]

    def normalize(self, method='scaling', lowerpercentile=None, upperpercentile=None):
        # Convert the attribute "data" to float()
        self.data = self.data.astype(np.float32)

        if lowerpercentile is not None:
            qlow = np.percentile(self.data, lowerpercentile)
        if upperpercentile is not None:
            qup = np.percentile(self.data, upperpercentile)

        if lowerpercentile is not None:
            self.data[self.data < qlow] = qlow
        if upperpercentile is not None:
            self.data[self.data > qup] = qup

        if method == 'scaling':
            # Divide "data" by its maximum value
            self.data -= self.data.min()
            self.data = np.multiply(self.data, 1.0 / self.data.max())
        elif method == 'standardization':
            self.data = self.data - np.mean(self.data)
            self.data = self.data / np.std(self.data)

    def get_slice(self, the_slice, axis='axial'):
        indices = [slice(None)] * self.data.ndim
        indices[NRRD.VIEW_MAPPING[axis]] = the_slice
        return self.data[indices]

    def get_data(self):
        return self.data

    def set_to_zero(self):
        self.data.fill(0.0)

    def visualize(self, axis='axial', pause=0.2):
        for i in range(self.data.shape[NRRD.VIEW_MAPPING[axis]]):
            img = self.get_slice(i, axis=axis)
            plt.imshow(img)
            plt.pause(pause)
