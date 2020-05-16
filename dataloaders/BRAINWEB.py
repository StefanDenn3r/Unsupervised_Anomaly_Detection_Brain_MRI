"""Functions for reading BRAINWEB NII data."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import glob
import math
import os.path
import pickle

import cv2
import matplotlib.pyplot
from imageio import imwrite
from scipy.ndimage import rotate

from utils.MINC import *
from utils.image_utils import crop, crop_center
from utils.tfrecord_utils import *


class BRAINWEB(object):
    FILTER_TYPES = ['NORMAL', 'MILDMS', 'MODERATEMS', 'SEVEREMS']
    SET_TYPES = ['TRAIN', 'VAL', 'TEST']
    LABELS = {'BACKGROUND': 0, 'CSF': 1, 'GM': 2, 'WM': 3, 'FAT': 4, 'MUSCLE': 5, 'SKIN': 6, 'SKULL': 7, 'GLIALMATTER': 8, 'CONNECTIVE': 9, 'LESION': 10}
    VIEW_MAPPING = {'saggital': 0, 'coronal': 1, 'axial': 2}
    PROTOCOL_MAPPINGS = {'FLAIR': 'flair*', 'T2': 't2*'}

    class Options(object):
        def __init__(self):
            self.description = None
            self.dir = os.path.dirname(os.path.realpath(__file__))
            self.folderNormal = 'normal'
            self.folderMildMS = os.path.join('lesions', 'mild')
            self.folderModerateMS = os.path.join('lesions', 'moderate')
            self.folderSevereMS = os.path.join('lesions', 'severe')
            self.folderGT = 'groundtruth'
            self.numSamples = -1
            self.partition = {'TRAIN': 0.6, 'VAL': 0.15, 'TEST': 0.25}
            self.sliceStart = 20
            self.sliceEnd = 140
            self.useCrops = False
            self.cropType = 'random'  # random or center
            self.numRandomCropsPerSlice = 5
            self.rotations = [0]
            self.cropWidth = 128
            self.cropHeight = 128
            self.cache = False
            self.sliceResolution = None  # format: HxW
            self.addInstanceNoise = False  # Affects only the batch sampling. If True, a tiny bit of noise will be added to every batch
            self.filterProtocol = None  # T2 or FLAIR only, not implemented for now
            self.filterType = None  # MILDMS, MODERATEMS, SEVEREMS, NORMAL
            self.axis = 'axial'  # saggital, coronal or axial
            self.debug = False
            self.normalizationMethod = 'standardization'
            self.skullRemoval = False
            self.backgroundRemoval = False

    def __init__(self, options=Options()):
        self.options = options

        if options.cache and os.path.isfile(self.pckl_name()):
            f = open(self.pckl_name(), 'rb')
            tmp = pickle.load(f)
            f.close()
            self._epochs_completed = tmp._epochs_completed
            self._index_in_epoch = tmp._index_in_epoch
            self.patients = self._get_patients()
            self._images, self._labels, self._sets = read_tf_record(self.tfrecord_name())

            f = open(self.split_name(), 'rb')
            self.patients_split = pickle.load(f)
            f.close()
            if not os.path.exists(self.split_name() + ".deprecated"):
                os.rename(self.split_name(), self.split_name() + ".deprecated")
            self._convert_patient_split()

            self._epochs_completed = {'TRAIN': 0, 'VAL': 0, 'TEST': 0}
            self._index_in_epoch = {'TRAIN': 0, 'VAL': 0, 'TEST': 0}
        else:
            # Collect all patients
            self.patients = self._get_patients()
            self.patients_split = {}  # Here we will later store the info whether a patient belongs to train, val or test

            # Determine Train, Val & Test set based on patients
            if not os.path.isfile(self.split_name()):
                _num_patients = len(self.patients)
                _ridx = numpy.random.permutation(_num_patients)

                _already_taken = 0
                for split in self.options.partition.keys():
                    if 1.0 >= self.options.partition[split] > 0.0:
                        num_patients_for_current_split = max(1, math.floor(self.options.partition[split] * _num_patients))
                    else:
                        num_patients_for_current_split = int(self.options.partition[split])

                    if num_patients_for_current_split > (_num_patients - _already_taken):
                        num_patients_for_current_split = _num_patients - _already_taken

                    self.patients_split[split] = _ridx[_already_taken:_already_taken + num_patients_for_current_split]
                    _already_taken += num_patients_for_current_split

                self._convert_patient_split()  # NEW! We have a new format for storing hte patientsSplit which is OS agnostic.
            else:
                f = open(self.split_name(), 'rb')
                self.patients_split = pickle.load(f)
                f.close()
                self._convert_patient_split()  # NEW! We have a new format for storing hte patientsSplit which is OS agnostic.

            # Iterate over all patients and the filtered NII files and extract slices
            _images = []
            _labels = []
            _sets = []
            for p, patient in enumerate(self.patients):
                if patient["name"] in self.patients_split['TRAIN']:
                    _set_of_current_patient = BRAINWEB.SET_TYPES.index('TRAIN')
                elif patient["name"] in self.patients_split['VAL']:
                    _set_of_current_patient = BRAINWEB.SET_TYPES.index('VAL')
                elif patient["name"] in self.patients_split['TEST']:
                    _set_of_current_patient = BRAINWEB.SET_TYPES.index('TEST')

                minc, minc_seg, minc_skullmap = self.load_volume_and_groundtruth(patient["filtered_files"][0], patient)

                # Iterate over all slices and collect them
                for s in range(self.options.sliceStart, min(self.options.sliceEnd, minc.num_slices_along_axis(self.options.axis))):
                    if 0 < self.options.numSamples < len(_images):
                        break

                    slice_data = minc.get_slice(s, self.options.axis)
                    slice_seg = minc_seg.get_slice(s, self.options.axis)

                    # Skip the slice if it is entirely black
                    if numpy.unique(slice_data).size == 1:
                        continue

                    # assert numpy.max(slice_data) <= 1.0, "Slice range is outside [0; 1]!"

                    if self.options.sliceResolution is not None:
                        # If the images are too big in resolution, do downsampling
                        if slice_data.shape[0] > self.options.sliceResolution[0] or slice_data.shape[1] > self.options.sliceResolution[1]:
                            slice_data = cv2.resize(slice_data, tuple(self.options.sliceResolution))
                            slice_seg = cv2.resize(slice_seg, tuple(self.options.sliceResolution), interpolation=cv2.INTER_NEAREST)
                        # Otherwise, do zero padding
                        else:
                            tmp_slice = numpy.zeros(self.options.sliceResolution)
                            tmp_slice_seg = numpy.zeros(self.options.sliceResolution)
                            start_x = (self.options.sliceResolution[1] - slice_data.shape[1]) // 2
                            start_y = (self.options.sliceResolution[0] - slice_data.shape[0]) // 2
                            end_x = start_x + slice_data.shape[1]
                            end_y = start_y + slice_data.shape[0]
                            tmp_slice[start_y:end_y, start_x:end_x] = slice_data
                            tmp_slice_seg[start_y:end_y, start_x:end_x] = slice_seg
                            slice_data = tmp_slice
                            slice_seg = tmp_slice_seg

                    for angle in self.options.rotations:
                        if angle != 0:
                            slice_data_rotated = rotate(slice_data, angle, reshape=False)
                            slice_seg_rotated = rotate(slice_seg, angle, reshape=False, mode='nearest')
                        else:
                            slice_data_rotated = slice_data
                            slice_seg_rotated = slice_seg

                        # Either collect crops
                        if self.options.useCrops:
                            if self.options.cropType == 'random':
                                rx = numpy.random.randint(0, high=(slice_data_rotated.shape[1] - self.options.cropWidth),
                                                          size=self.options.numRandomCropsPerSlice)
                                ry = numpy.random.randint(0, high=(slice_data_rotated.shape[0] - self.options.cropHeight),
                                                          size=self.options.numRandomCropsPerSlice)
                                for r in range(self.options.numRandomCropsPerSlice):
                                    _images.append(crop(slice_data_rotated, ry[r], rx[r], self.options.cropHeight, self.options.cropWidth))
                                    _labels.append(crop(slice_data_rotated, ry[r], rx[r], self.options.cropHeight, self.options.cropWidth))
                                    _sets.append(_set_of_current_patient)
                            elif self.options.cropType == 'center':
                                slice_data_cropped = crop_center(slice_data_rotated, self.options.cropWidth, self.options.cropHeight)
                                slice_seg_cropped = crop_center(slice_seg_rotated, self.options.cropWidth, self.options.cropHeight)
                                _images.append(slice_data_cropped)
                                _labels.append(slice_seg_cropped)
                                _sets.append(_set_of_current_patient)
                        # Or whole slices
                        else:
                            _images.append(slice_data_rotated)
                            _labels.append(slice_seg_rotated)
                            _sets.append(_set_of_current_patient)

            self._images = numpy.array(_images).astype(numpy.float32)
            self._labels = numpy.array(_labels).astype(numpy.float32)
            # assert numpy.max(self._images) <= 1.0, "MINC range is outside [0; 1]!"
            if self._images.ndim < 4:
                self._images = numpy.expand_dims(self._images, 3)
            self._sets = numpy.array(_sets).astype(numpy.int32)
            self._epochs_completed = {'TRAIN': 0, 'VAL': 0, 'TEST': 0}
            self._index_in_epoch = {'TRAIN': 0, 'VAL': 0, 'TEST': 0}

            if self.options.cache:
                write_tf_record(self._images, self._labels, self._sets, self.tfrecord_name())
                tmp = copy.copy(self)
                tmp._images = None
                tmp._labels = None
                tmp._sets = None
                f = open(self.pckl_name(), 'wb')
                pickle.dump(tmp, f)
                f.close()

    def _get_patients(self):
        return BRAINWEB.get_patients(self.options)

    @staticmethod
    def get_patients(options):
        minc_folders = [options.folderNormal, options.folderMildMS, options.folderModerateMS, options.folderSevereMS]

        # Iterate over all folders and collect patients
        patients = []
        for n, minc_folder in enumerate(minc_folders):
            if minc_folder == options.folderNormal:
                _type = 'NORMAL'
            elif minc_folder == options.folderMildMS:
                _type = 'MILDMS'
            elif minc_folder == options.folderModerateMS:
                _type = 'MODERATEMS'
            elif minc_folder == options.folderSevereMS:
                _type = 'SEVEREMS'

            # Continue with the next patient if the current one is not part of the desired types
            if _type not in options.filterType:
                continue

            if options.filterProtocol:
                _regex = BRAINWEB.PROTOCOL_MAPPINGS[options.filterProtocol] + ".mnc.gz"
            else:
                _regex = "*.mnc.gz"
            _files = glob.glob(os.path.join(options.dir, minc_folder, _regex))
            for f, fname in enumerate(_files):
                patient = {
                    'name': os.path.basename(fname),
                    'type': _type,
                    'fullpath': fname
                }
                patient['filtered_files'] = patient['fullpath']

                if patient['type'] == 'NORMAL':
                    patient['groundtruth_filename'] = os.path.join(options.dir, options.folderGT, 'normal.mnc.gz')
                elif patient['type'] == 'MILDMS':
                    patient['groundtruth_filename'] = os.path.join(options.dir, options.folderGT, 'mild_lesions.mnc.gz')
                elif patient['type'] == 'MODERATEMS':
                    patient['groundtruth_filename'] = os.path.join(options.dir, options.folderGT, 'moderate_lesions.mnc.gz')
                elif patient['type'] == 'SEVEREMS':
                    patient['groundtruth_filename'] = os.path.join(options.dir, options.folderGT, 'severe_lesions.mnc.gz')

                patients.append(patient)

        return patients

    def load_volume_and_groundtruth(self, minc_filename, patient):
        minc_filename = patient['fullpath']
        try:
            minc = MINC(minc_filename)  # NII also works with MINC
            minc.set_view_mapping(BRAINWEB.VIEW_MAPPING)
        except:
            print('BRAINWEB: Failed to open file ' + minc_filename)

        # Try to load the segmentation ground-truth
        minc_seg_path = patient["groundtruth_filename"]
        minc_seg = MINC(minc_seg_path)
        skullmap = MINC(minc_seg_path)
        skullmap.data = (skullmap.data * 0.0) + 1.0
        skullmap.set_view_mapping(BRAINWEB.VIEW_MAPPING)
        minc_seg.set_view_mapping(BRAINWEB.VIEW_MAPPING)

        # If desired, compute the skullmap
        if self.options.skullRemoval:
            skullmap.data[minc_seg.data == BRAINWEB.LABELS['FAT']] = 0
            skullmap.data[minc_seg.data == BRAINWEB.LABELS['MUSCLE']] = 0
            skullmap.data[minc_seg.data == BRAINWEB.LABELS['SKIN']] = 0
            skullmap.data[minc_seg.data == BRAINWEB.LABELS['SKULL']] = 0
            skullmap.data[minc_seg.data == BRAINWEB.LABELS['CONNECTIVE']] = 0

        if self.options.backgroundRemoval:
            skullmap.data[minc_seg.data == BRAINWEB.LABELS['BACKGROUND']] = 0

        # Binarize minc_seg
        lesion_idx = (minc_seg.data == BRAINWEB.LABELS['LESION'])
        nonlesion_idx = (minc_seg.data != BRAINWEB.LABELS['LESION'])
        minc_seg.data[lesion_idx] = 1
        minc_seg.data[nonlesion_idx] = 0

        if self.options.skullRemoval or self.options.backgroundRemoval:
            minc.apply_skullmap(skullmap)

        # In-place normalize the loaded volume
        minc.normalize(method=self.options.normalizationMethod, lowerpercentile=0.0, upperpercentile=99.8)
        # 99.8 percentile described in LG Ny´ul, Jayaram K Udupa, and Xuan Zhang.
        # New variants of a method of MRI scale standardization.
        # IEEE transactions on medical imaging, 19(2):143–150, 2000.
        # assert numpy.max(minc.getData()) <= 1.0, "MINC range is outside [0; 1]!"

        return minc, minc_seg, skullmap

    # Returns the indices of patients which belong to either TRAIN, VAL or TEST. Your choice
    def get_patient_idx(self, split='TRAIN'):
        idx = []
        for pidx, patient in enumerate(self.patients):
            if patient["name"] in self.patients_split[split]:
                idx += [pidx]
        return idx

    def get_patient_split(self):
        return self.patients_split

    @property
    def images(self):
        return self._images

    def get_images(self, set=None):
        _setIdx = BRAINWEB.SET_TYPES.index(set)
        images_in_set = numpy.where(self._sets == _setIdx)[0]
        return self._images[images_in_set]

    def get_image(self, i):
        return self._images[i, :, :, :]

    def get_label(self, i):
        return self._labels[i, :, :, :]

    @property
    def labels(self):
        return self._labels

    @property
    def sets(self):
        return self._sets

    @property
    def meta(self):
        return self._meta

    @property
    def num_examples(self):
        return self._images.shape[0]

    @property
    def width(self):
        return self._images.shape[2]

    @property
    def height(self):
        return self._images.shape[1]

    @property
    def num_channels(self):
        return self._images.shape[3]

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def name(self):
        _name = "BRAINWEB"
        if self.options.description:
            _name += "_{}".format(self.options.description)
        if self.options.numSamples > 0:
            _name += '_n{}'.format(self.options.numSamples)
        _name += "_p{}-{}-{}".format(self.options.partition['TRAIN'], self.options.partition['VAL'], self.options.partition['TEST'])
        if self.options.useCrops:
            _name += "_{}crops{}x{}".format(self.options.cropType, self.options.cropWidth, self.options.cropHeight)
            if self.options.cropType == "random":
                _name += "_{}cropsPerSlice".format(self.options.numRandomCropsPerSlice)
        if self.options.sliceResolution is not None:
            _name += "_res{}x{}".format(self.options.sliceResolution[0], self.options.sliceResolution[1])
        if self.options.skullRemoval:
            _name += "_noSkull"
        if self.options.backgroundRemoval:
            _name += "_noBackground"
        return _name

    def pckl_name(self):
        return os.path.join(self.dir(), self.name() + ".pckl")

    def tfrecord_name(self):
        return os.path.join(self.dir(), self.name() + ".tfrecord")

    def split_name(self):
        return os.path.join(self.dir(),
                            'split-{}-{}-{}.pckl'.format(self.options.partition['TRAIN'], self.options.partition['VAL'], self.options.partition['TEST']))

    def dir(self):
        return self.options.dir

    def export_slices(self, dir):
        for i in range(self.num_examples):
            imwrite(os.path.join(dir, '{}.png'.format(i)), np.squeeze(self.get_image(i) * 255).astype('uint8'))

    def visualize(self, pause=1, set='TRAIN'):
        f, (ax1, ax2) = matplotlib.pyplot.subplots(1, 2)
        images_tmp, labels_tmp, _ = self.next_batch(10, set=set)
        for i in range(images_tmp.shape[0]):
            img = numpy.squeeze(images_tmp[i])
            lbl = numpy.squeeze(labels_tmp[i])
            ax1.imshow(img)
            ax1.set_title('Patch')
            ax2.imshow(lbl)
            ax2.set_title('Groundtruth')
            matplotlib.pyplot.pause(pause)

    def num_batches(self, batchsize, set='TRAIN'):
        _setIdx = BRAINWEB.SET_TYPES.index(set)
        images_in_set = numpy.where(self._sets == _setIdx)[0]
        return len(images_in_set) // batchsize

    def next_batch(self, batch_size, shuffle=True, set='TRAIN', return_brainmask=False):
        """Return the next `batch_size` examples from this data set."""
        _setIdx = BRAINWEB.SET_TYPES.index(set)
        images_in_set = numpy.where(self._sets == _setIdx)[0]
        samples_in_set = len(images_in_set)

        start = self._index_in_epoch[set]
        # Shuffle for the first epoch
        if self._epochs_completed == 0 and start == 0 and shuffle:
            perm0 = numpy.arange(samples_in_set)
            numpy.random.shuffle(perm0)
            self._images[images_in_set] = self.images[images_in_set[perm0]]
            self._labels[images_in_set] = self.labels[images_in_set[perm0]]
            self._sets[images_in_set] = self.sets[images_in_set[perm0]]

        # Go to the next epoch
        if start + batch_size > samples_in_set:
            # Finished epoch
            self._epochs_completed[set] += 1

            # Get the rest examples in this epoch
            rest_num_examples = samples_in_set - start
            images_rest_part = self._images[images_in_set[start:samples_in_set]]
            labels_rest_part = self._labels[images_in_set[start:samples_in_set]]

            # Shuffle the data
            if shuffle:
                perm = numpy.arange(samples_in_set)
                numpy.random.shuffle(perm)
                self._images[images_in_set] = self.images[images_in_set[perm]]
                self._labels[images_in_set] = self.labels[images_in_set[perm]]
                self._sets[images_in_set] = self.sets[images_in_set[perm]]

            # Start next epoch
            start = 0
            self._index_in_epoch[set] = batch_size - rest_num_examples
            end = self._index_in_epoch[set]
            images_new_part = self._images[images_in_set[start:end]]
            labels_new_part = self._labels[images_in_set[start:end]]

            images_tmp = numpy.concatenate((images_rest_part, images_new_part), axis=0)
            labels_tmp = numpy.concatenate((labels_rest_part, labels_new_part), axis=0)
        else:
            self._index_in_epoch[set] += batch_size
            end = self._index_in_epoch[set]
            images_tmp = self._images[images_in_set[start:end]]
            labels_tmp = self._labels[images_in_set[start:end]]

        if self.options.addInstanceNoise:
            noise = numpy.random.normal(0, 0.01, images_tmp.shape)
            images_tmp += noise

        # Check the batch
        assert images_tmp.size, "The batch is empty!"
        assert labels_tmp.size, "The labels of the current batch are empty!"

        if return_brainmask:
            brainmasks = np.copy(labels_tmp)
            brainmasks[brainmasks == BRAINWEB.LABELS['FAT']] = 0
            brainmasks[brainmasks == BRAINWEB.LABELS['MUSCLE']] = 0
            brainmasks[brainmasks == BRAINWEB.LABELS['SKIN']] = 0
            brainmasks[brainmasks == BRAINWEB.LABELS['SKULL']] = 0
            brainmasks[brainmasks == BRAINWEB.LABELS['CONNECTIVE']] = 0
            brainmasks[brainmasks == BRAINWEB.LABELS['BACKGROUND']] = 0
            brainmasks[brainmasks > 0] = 1
            return images_tmp, labels_tmp, brainmasks

        return images_tmp, labels_tmp, None

    def _convert_patient_split(self):
        for split in self.patients_split.keys():
            _list_of_patient_names = []
            for pidx in self.patients_split[split]:
                if not isinstance(pidx, str):
                    _list_of_patient_names += [self.patients[pidx]['name']]
                else:
                    _list_of_patient_names = self.patients_split[split]
                    break
            self.patients_split[split] = _list_of_patient_names

        f = open(self.split_name(), 'wb')
        pickle.dump(self.patients_split, f)
        f.close()
