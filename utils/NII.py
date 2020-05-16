import copy

import SimpleITK as sitk
import matplotlib.pyplot as plt
import numpy as np


class NII:
    VIEW_MAPPING = {'saggital': 0, 'coronal': 1, 'axial': 2}

    def __init__(self, filename):
        self.nii = sitk.ReadImage(filename, sitk.sitkFloat64)
        self._update_attributes()

        # Remove NaNs
        self.data[np.isnan(self.data)] = 0

    def update_sitk(self):
        self.nii = sitk.GetImageFromArray(self.data)
        self.nii.SetOrigin(self.origin)
        self.nii.SetDirection(self.direction)

    def _update_attributes(self):
        self.origin = self.nii.GetOrigin()
        self.direction = self.nii.GetDirection()
        self.data = sitk.GetArrayFromImage(self.nii)

    def save(self, filename):
        sitk.WriteImage(self.nii, filename)

    @property
    def num_saggital_slices(self):
        return self.data.shape[NII.VIEW_MAPPING['saggital']]

    @property
    def num_coronal_slices(self):
        return self.data.shape[NII.VIEW_MAPPING['coronal']]

    @property
    def num_axial_slices(self):
        return self.data.shape[NII.VIEW_MAPPING['axial']]

    @staticmethod
    def set_view_mapping(mapping):
        NII.VIEW_MAPPING = mapping

    def shape(self):
        return self.data.shape

    def num_slices_along_axis(self, axis):
        return self.data.shape[NII.VIEW_MAPPING[axis]]

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
            if self.data.max() > 0.0:
                self.data = np.multiply(self.data, 1.0 / self.data.max())
        elif method == 'standardization':
            self.data = self.data - np.mean(self.data)
            self.data = self.data / np.std(self.data)

        self.update_sitk()

    def apply_skullmap(self, skullmap):
        brainmask = skullmap.get_data()
        brainmask[brainmask < 0.1] = 0
        brainmask[brainmask >= 0.1] = 1
        self.data = self.data * brainmask

        self.update_sitk()

    def denoise(self):
        self.nii = sitk.CurvatureFlow(image1=self.nii, timeStep=0.125, numberOfIterations=3)
        self._update_attributes()

    def subtract(self, filename):
        nii_sub = NII(filename)
        self.data = self.data - nii_sub.get_data()
        self.update_sitk()

    def get_slice(self, the_slice, axis='axial'):
        indices = [slice(None)] * self.data.ndim
        indices[NII.VIEW_MAPPING[axis]] = the_slice
        return self.data[tuple(indices)]

    def set_slice(self, the_slice, the_data, axis='axial'):
        indices = [slice(None)] * self.data.ndim
        indices[NII.VIEW_MAPPING[axis]] = the_slice
        self.data[tuple(indices)] = the_data
        self.update_sitk()
        self._update_attributes()

    # The first index of the subvolume is expected to be the axis we iterate over
    def set_subvolume(self, slice_start, slice_end, subvolume, axis='axial'):
        for s in range(slice_start, slice_end):
            self.set_slice(s, subvolume[s - slice_start, :, :], axis)

    def get_data(self):
        return self.data

    def cast_to_float(self):
        self.nii = sitk.Cast(self.nii, sitk.sitkFloat64)
        self._update_attributes()

    def set_to_zero(self):
        self.data.fill(0.0)
        self.update_sitk()

    def visualize(self, axis='axial', pause=0.2):
        num_slices = self.data.shape[NII.VIEW_MAPPING[axis]]
        for i in range(num_slices):
            img = self.get_slice(i, axis=axis)
            plt.imshow(img)
            plt.title(f"Slice {i}/{num_slices}")
            plt.pause(pause)
            plt.cla()

    def copy(self):
        return copy.deepcopy(self)
