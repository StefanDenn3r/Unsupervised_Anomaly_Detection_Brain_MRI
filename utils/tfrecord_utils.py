import numpy
import tensorflow as tf


# Functions
def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def write_tf_record(images, labels, sets, filename):
    writer = tf.python_io.TFRecordWriter(filename)
    for i in range(0, images.shape[0]):
        img = images[i]
        label = labels[i]
        set = sets[i]
        img_raw = img.tostring()
        label_raw = label.tostring()
        set_raw = set.tostring()

        example = tf.train.Example(features=tf.train.Features(feature={
            'height': _int64_feature(img.shape[0]),
            'width': _int64_feature(img.shape[1]),
            'image': _bytes_feature(img_raw),
            'label': _bytes_feature(label_raw),
            'set': _bytes_feature(set_raw)})
        )

        writer.write(example.SerializeToString())
    writer.close()


def read_tf_record(filename):
    images = []
    labels = []
    sets = []
    record_iterator = tf.python_io.tf_record_iterator(path=filename)
    for string_record in record_iterator:
        example = tf.train.Example()
        example.ParseFromString(string_record)
        height = int(example.features.feature["height"].int64_list.value[0])
        width = int(example.features.feature["width"].int64_list.value[0])
        img_string = (example.features.feature["image"].bytes_list.value[0])
        label_string = (example.features.feature["label"].bytes_list.value[0])
        set_string = (example.features.feature["set"].bytes_list.value[0])
        images.append(numpy.fromstring(img_string, dtype=numpy.float32).reshape(height, width, -1))
        labels.append(numpy.fromstring(label_string, dtype=numpy.float32).reshape(height, width, -1))
        sets.append(numpy.fromstring(set_string, dtype=numpy.int32))
    return numpy.array(images), numpy.array(labels), numpy.array(sets)
