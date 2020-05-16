"""Functions for reading MSSEG2008 NRRD data."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os.path
import pickle

import matplotlib.pyplot
from imageio import imwrite
from scipy.ndimage import zoom
from six.moves import xrange  # pylint: disable=redefined-builtin
from skimage.measure import label, regionprops

from dataloaders.NRRD import *
from utils.NII import *
from utils.image_utils import crop, crop_center
from utils.tfrecord_utils import *


class MSSEG2008(object):
    PROTOCOL_MAPPINGS = ['FLAIR', 'T1', 'T2']
    SET_TYPES = ['TRAIN', 'VAL', 'TEST']

    class Options(object):
        def __init__(self):
            self.dir = os.path.dirname(os.path.realpath(__file__))
            self.folderTrainUNC = 'UNC_train'
            self.folderTestUNC = 'UNC_test'
            self.folderTrainCHB = 'CHB_train'
            self.folderTestCHB = 'CHB_test'
            self.numSamples = -1
            self.partition = {'TRAIN': 0.7, 'VAL': 0.2, 'TEST': 0.1}
            self.useCrops = False
            self.cropType = 'random'  # random or center
            self.numRandomCropsPerSlice = 5
            self.onlyPatchesWithLesions = False
            self.rotations = 0
            self.cropWidth = 128
            self.cropHeight = 128
            self.cache = False
            self.sliceResolution = None  # format: HxW
            self.addInstanceNoise = False  # Affects only the batch sampling. If True, a tiny bit of noise will be added to every batch
            self.filterProtocol = None  # FLAIR, T1, T2
            self.filterScanner = "UNC"  # UNC or CHB
            self.filterType = "train"  # train or test
            self.axis = 'axial'  # saggital, coronal or axial
            self.debug = False
            self.normalizationMethod = 'standardization'
            self.sliceStart = 0
            self.sliceEnd = 155
            self.format = "raw"  # raw or aligned; If aligned, nii-files will be crawled and loaded
            self.skullStripping = True
            self.viewMapping = {'saggital': 2, 'coronal': 1, 'axial': 0}

    def __init__(self, options=Options()):
        self.options = options

        if options.cache and os.path.isfile(self.pckl_name()):
            f = open(self.pckl_name(), 'rb')
            tmp = pickle.load(f)
            f.close()
            self._epochs_completed = tmp._epochs_completed
            self._index_in_epoch = tmp._index_in_epoch
            self.patientsSplit = tmp.patients_split
            self.patients = tmp.patients
            self._images, self._labels, self._sets = read_tf_record(self.tfrecord_name())
            self._epochs_completed = {'TRAIN': 0, 'VAL': 0, 'TEST': 0}
            self._index_in_epoch = {'TRAIN': 0, 'VAL': 0, 'TEST': 0}
        else:
            # Collect all patients
            self.patients = self._get_patients()
            self.patientsSplit = {}

            if not os.path.isfile(self.split_name()):
                _numPatients = len(self.patients)
                _ridx = numpy.random.permutation(_numPatients)

                _already_taken = 0
                for split in self.options.partition.keys():
                    if self.options.partition[split] <= 1.0:
                        numPatientsForCurrentSplit = math.floor(self.options.partition[split] * _numPatients)
                    else:
                        numPatientsForCurrentSplit = self.options.partition[split]

                    if numPatientsForCurrentSplit > (_numPatients - _already_taken):
                        numPatientsForCurrentSplit = _numPatients - _already_taken

                    self.patientsSplit[split] = _ridx[_already_taken:_already_taken + numPatientsForCurrentSplit]
                    _already_taken += numPatientsForCurrentSplit

                f = open(self.split_name(), 'wb')
                pickle.dump(self.patientsSplit, f)
                f.close()
            else:
                f = open(self.split_name(), 'rb')
                self.patientsSplit = pickle.load(f)
                f.close()

            self._create_numpy_arrays()

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

    def _create_numpy_arrays(self):
        # Iterate over all patients and extract slices
        _images = []
        _labels = []
        _sets = []
        for p, patient in enumerate(self.patients):
            if p in self.patientsSplit['TRAIN']:
                _set_of_current_patient = MSSEG2008.SET_TYPES.index('TRAIN')
            elif p in self.patientsSplit['VAL']:
                _set_of_current_patient = MSSEG2008.SET_TYPES.index('VAL')
            elif p in self.patientsSplit['TEST']:
                _set_of_current_patient = MSSEG2008.SET_TYPES.index('TEST')

            for n, nrrd_filename in enumerate(patient['filtered_files']):
                # try:
                _images_tmp, _labels_tmp = self.gather_data(patient, nrrd_filename)
                _images += _images_tmp
                _labels += _labels_tmp
                # _mask += _mask_tmp
                _sets += [_set_of_current_patient] * len(_images_tmp)
                # except:
                #  print('MSSEG2008: Failed to open file ' + nrrd_filename)
                #  continue

        self._images = numpy.array(_images).astype(numpy.float32)
        self._labels = numpy.array(_labels).astype(numpy.float32)
        if self._images.ndim < 4:
            self._images = numpy.expand_dims(self._images, 3)
        self._sets = numpy.array(_sets).astype(numpy.int32)

    def gather_data(self, patient, nrrd_filename):
        _images = []
        _labels = []

        nrrd, nrrd_seg, nrrd_skullmap = self.load_volume_and_groundtruth(nrrd_filename, patient)

        # Iterate over all slices and collect them
        # We only want to select in the range from 15 to 125 (in axial view)
        for s in xrange(self.options.sliceStart, min(self.options.sliceEnd, nrrd.num_slices_along_axis(self.options.axis))):
            if 0 < self.options.numSamples < len(_images):
                break

            slice_data = nrrd.get_slice(s, self.options.axis)
            slice_seg = nrrd_seg.get_slice(s, self.options.axis)
            slice_skullmap = nrrd_skullmap.get_slice(s, self.options.axis)

            # Skip the slice if it is "empty"
            # if numpy.max(slice_data) < empty_thresh:
            if numpy.percentile(slice_data, 90) < 0.2:
                continue

            # assert numpy.max(slice_data) <= 1.0, "Slice range is outside [0; 1]!"

            if self.options.sliceResolution is not None:
                # Pad withzeros to top and bottom, if the image is too small
                if slice_data.shape[0] < self.options.sliceResolution[0]:
                    before_y = math.floor((self.options.sliceResolution[0] - slice_data.shape[0]) / 2.0)
                    after_y = math.ceil((self.options.sliceResolution[0] - slice_data.shape[0]) / 2.0)
                if slice_data.shape[1] < self.options.sliceResolution[1]:
                    before_x = math.floor((self.options.sliceResolution[1] - slice_data.shape[1]) / 2.0)
                    after_x = math.ceil((self.options.sliceResolution[1] - slice_data.shape[1]) / 2.0)
                if slice_data.shape[0] < self.options.sliceResolution[0] or slice_data.shape[1] < self.options.sliceResolution[1]:
                    slice_data = np.pad(slice_data, ((before_y, after_y), (before_x, after_x)), 'constant', constant_values=(0, 0))
                    slice_seg = np.pad(slice_seg, ((before_y, after_y), (before_x, after_x)), 'constant', constant_values=(0, 0))
                slice_data = zoom(slice_data, float(self.options.sliceResolution[0]) / float(slice_data.shape[0]))
                slice_seg = zoom(slice_seg, float(self.options.sliceResolution[0]) / float(slice_seg.shape[0]), mode="nearest")
                slice_seg[slice_seg < 0.9] = 0.0
                slice_seg[slice_seg >= 0.9] = 1.0

            # Either collect crops
            if self.options.useCrops:
                if self.options.cropType == 'random':
                    rx = numpy.random.randint(0, high=(slice_data.shape[1] - self.options.cropWidth),
                                              size=self.options.numRandomCropsPerSlice)
                    ry = numpy.random.randint(0, high=(slice_data.shape[0] - self.options.cropHeight),
                                              size=self.options.numRandomCropsPerSlice)
                    for r in range(self.options.numRandomCropsPerSlice):
                        _images.append(crop(slice_data, ry(r), rx(r), self.options.cropHeight, self.options.cropWidth))
                        _labels.append(crop(slice_data, ry(r), rx(r), self.options.cropHeight, self.options.cropWidth))
                elif self.options.cropType == 'center':
                    slice_data_cropped = crop_center(slice_data, self.options.cropWidth, self.options.cropHeight)
                    slice_seg_cropped = crop_center(slice_seg, self.options.cropWidth, self.options.cropHeight)
                    _images.append(slice_data_cropped)
                    _labels.append(slice_seg_cropped)
                elif self.options.cropType == 'lesions':
                    cc_slice = label(slice_seg)
                    props = regionprops(cc_slice)
                    if len(props) > 0:
                        for prop in props:
                            cx = prop['centroid'][1]
                            cy = prop['centroid'][0]
                            if cy < self.options.cropHeight // 2:
                                cy = self.options.cropHeight // 2
                            if cy > (slice_data.shape[0] - (self.options.cropHeight // 2)):
                                cy = (slice_data.shape[0] - (self.options.cropHeight // 2))
                            if cx < self.options.cropWidth // 2:
                                cx = self.options.cropWidth // 2
                            if cx > (slice_data.shape[1] - (self.options.cropWidth // 2)):
                                cx = (slice_data.shape[1] - (self.options.cropWidth // 2))
                            image_crop = crop(slice_data, int(cy) - (self.options.cropHeight // 2), int(cx) - (self.options.cropWidth // 2),
                                              self.options.cropHeight, self.options.cropWidth)
                            seg_crop = crop(slice_seg, int(cy) - (self.options.cropHeight // 2), int(cx) - (self.options.cropWidth // 2),
                                            self.options.cropHeight, self.options.cropWidth)
                            if image_crop.shape[0] != self.options.cropHeight or image_crop.shape[1] != self.options.cropWidth:
                                continue
                            _images.append(image_crop)
                            _labels.append(seg_crop)
                            # _masks.append(crop(slice_data, prop['centroid'][0], prop['centroid'][1], self.options.cropHeight, self.options.cropWidth))
                        # find connected components in segmentation slice
                        # for every connected component, do a center crop from the segmentation slice, the mask and the actual slice
            # Or whole slices
            else:
                _images.append(slice_data)
                _labels.append(slice_seg)

        return _images, _labels

    def load_volume_and_groundtruth(self, nrrd_filename, patient):
        # Load the nrrd
        try:
            if self.options.format == "raw":
                nrrd = NRRD(nrrd_filename)
                nrrd_groundtruth = NRRD(patient['groundtruth'])

                nrrd.denoise()
                nrrd.set_view_mapping(self.options.viewMapping)
            elif self.options.format == "aligned":
                nrrd = NII(nrrd_filename)
                nrrd_groundtruth = NII(patient['groundtruth'])
                nrrd.denoise()
                nrrd.set_view_mapping(self.options.viewMapping)
        except:
            print('MSSEG2008: Failed to open file ' + nrrd_filename)

        # Make sure ground-truth is binary and nrrd doesnt have NaNs
        nrrd.data[np.isnan(nrrd.data)] = 0.0
        nrrd_groundtruth.data[nrrd_groundtruth.data < 0.9] = 0.0
        nrrd_groundtruth.data[nrrd_groundtruth.data >= 0.9] = 1.0

        # Do skull-stripping, if desired
        if self.options.skullStripping:
            try:
                nii_skullmap = NII(patient['skullmap'])
                nii_skullmap.set_view_mapping(self.options.viewMapping)
                nrrd.apply_skullmap(nii_skullmap)
            except:
                print('MSSEG2008: Failed to open file ' + patient['skullmap'] + ', skipping skullremoval')

        # In-place normalize the loaded volume
        nrrd.normalize(method=self.options.normalizationMethod, lowerpercentile=0, upperpercentile=99.8)
        # nrrd_skullmap.data = nrrd_skullmap.data > 0.0

        return nrrd, nrrd_groundtruth, nii_skullmap

    # Hidden helper function, not supposed to be called from outside!
    def _get_patients(self):
        return MSSEG2008.get_patients(self.options)

    @staticmethod
    def get_patients(options):
        folders = [options.folderTrainUNC, options.folderTestUNC, options.folderTrainCHB, options.folderTestCHB]

        # Iterate over all folderHC, folderNC, folderPC and collect patients
        patients = []
        for f, folder in enumerate(folders):
            if options.filterScanner and options.filterScanner not in folder:
                continue
            if options.filterType and options.filterType not in folder:
                continue

            # Get all files that can be used for training and validation
            _patients = [f.name for f in os.scandir(os.path.join(options.dir, folder)) if f.is_dir()]
            for p, pname in enumerate(_patients):
                patient = {
                    'name': pname,
                    'fullpath': os.path.join(options.dir, folder, pname)
                }
                if "train" in folder:
                    patient["type"] = "train"
                else:
                    patient["type"] = "test"

                patient["filtered_files"] = []
                for pr, protocol in enumerate(MSSEG2008.PROTOCOL_MAPPINGS):
                    if options.format == "raw":
                        patient[protocol] = os.path.join(options.dir, folder, pname, pname + '_' + protocol + '.nhdr')
                    elif options.format == "aligned":
                        patient[protocol] = os.path.join(options.dir, folder, pname, pname + '_' + protocol + '.aligned.nii.gz')

                    if len(options.filterProtocols) > 0 and protocol not in options.filterProtocols:
                        continue
                    else:
                        if options.format == "raw":
                            patient["filtered_files"] += [os.path.join(options.dir, folder, pname, pname + '_' + protocol + '.nhdr')]
                        elif options.format == "aligned":
                            patient["filtered_files"] += [os.path.join(options.dir, folder, pname, pname + '_' + protocol + '.aligned.nii.gz')]

                if options.format == "raw":
                    patient['groundtruth'] = os.path.join(options.dir, folder, pname, pname + '_lesion.nhdr')
                    patient['skullmap'] = os.path.join(options.dir, folder, pname, pname + '_skullmap.nhdr')
                elif options.format == "aligned":
                    patient['groundtruth'] = os.path.join(options.dir, folder, pname, pname + '_lesion.aligned.nii.gz')
                    patient['skullmap'] = os.path.join(options.dir, folder, pname, pname + '_skullmap.nii.gz')

                # Append to the list of all patients
                patients.append(patient)

        return patients

    # Returns the indices of patients which belong to either TRAIN, VAL or TEST. Your choice
    def get_patient_idx(self, split='TRAIN'):
        return self.patientsSplit[split]

    def get_patient_split(self):
        return self.patientsSplit

    @property
    def images(self):
        return self._images

    def get_images(self, set=None):
        _setIdx = self.SET_TYPES.index(set)
        images_in_set = numpy.where(self._sets == _setIdx)[0]
        return self._images[images_in_set]

    def get_image(self, i):
        return self._images[i, :, :, :]

    def get_label(self, i):
        return self._labels[i, :, :, :]

    def get_patient(self, i):
        return self.patients[i]

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
        _name = "MSSEG2008"
        if self.options.filterScanner:
            _name += self.options.filterScanner
        if self.options.numSamples > 0:
            _name += '_n{}'.format(self.options.numSamples)
        _name += "_p{}-{}".format(self.options.partition['TRAIN'], self.options.partition['VAL'])
        if self.options.useCrops:
            _name += "_{}crops{}x{}".format(self.options.cropType, self.options.cropWidth, self.options.cropHeight)
            if self.options.cropType == "random":
                _name += "_{}cropsPerSlice".format(self.options.numRandomCropsPerSlice)
        if self.options.sliceResolution is not None:
            _name += "_res{}x{}".format(self.options.sliceResolution[0], self.options.sliceResolution[1])
        _name += "_{}".format(self.options.format)
        return _name

    def split_name(self):
        return os.path.join(self.dir(), 'split-{}-{}.pckl'.format(self.options.partition['TRAIN'], self.options.partition['VAL']))

    def pckl_name(self):
        return os.path.join(self.dir(), self.name() + ".pckl")

    def tfrecord_name(self):
        return os.path.join(self.dir(), self.name() + ".tfrecord")

    def dir(self):
        return self.options.dir

    def export_slices(self, dir):
        for i in range(self.num_examples):
            imwrite(os.path.join(dir, '{}.png'.format(i)), np.squeeze(self.get_image(i) * 255).astype('uint8'))

    def visualize(self, pause=1):
        f, (ax1, ax2) = matplotlib.pyplot.subplots(1, 2)
        images_tmp, labels_tmp, _ = self.next_batch(10)
        for i in range(images_tmp.shape[0]):
            img = numpy.squeeze(images_tmp[i])
            lbl = numpy.squeeze(labels_tmp[i])
            ax1.imshow(img)
            ax1.set_title('Patch')
            ax2.imshow(lbl)
            ax2.set_title('Groundtruth')
            matplotlib.pyplot.pause(pause)

    def num_batches(self, batchsize, set='TRAIN'):
        _setIdx = MSSEG2008.SET_TYPES.index(set)
        images_in_set = numpy.where(self._sets == _setIdx)[0]
        return len(images_in_set) // batchsize

    def next_batch(self, batch_size, shuffle=True, set='TRAIN', return_brainmask=True):
        """Return the next `batch_size` examples from this data set."""
        _setIdx = MSSEG2008.SET_TYPES.index(set)
        images_in_set = numpy.where(self._sets == _setIdx)[0]
        samples_in_set = len(images_in_set)

        start = self._index_in_epoch[set]
        # Shuffle for the first epoch
        if self._epochs_completed[set] == 0 and start == 0 and shuffle:
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
            brainmasks = images_tmp > 0.05
        else:
            brainmasks = None

        return images_tmp, labels_tmp, brainmasks
