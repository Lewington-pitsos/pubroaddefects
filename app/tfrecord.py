import tensorflow as tf
import random

from app.glob import *

parse_format = {
    TF_CLASS_LABEL: tf.io.FixedLenFeature([], tf.string),
    TF_OBJECT_LABEL: tf.io.FixedLenFeature([], tf.string),
    TF_COUNT_LABEL: tf.io.FixedLenFeature([], tf.string),
    TF_XMINS_LABEL: tf.io.FixedLenFeature([], tf.string),
    TF_XMAXS_LABEL: tf.io.FixedLenFeature([], tf.string),
    TF_YMINS_LABEL: tf.io.FixedLenFeature([], tf.string),
    TF_YMAXS_LABEL: tf.io.FixedLenFeature([], tf.string),

    TF_DATASET_NAME: tf.io.FixedLenFeature([], tf.string),
    TF_FILENAME: tf.io.FixedLenFeature([], tf.string),
    
    TF_IMAGE: tf.io.FixedLenFeature([], tf.string),
}

def write_img(img, img_data, writer):
    def _bytes_feature(value):
        if isinstance(value, type(tf.constant(0))):
            value = value.numpy() 
        if isinstance(value, str):
            value = bytes(value, 'utf-8')
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
    
    def _int64_feature(value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

    feature = {
        TF_IMAGE: _bytes_feature(tf.io.serialize_tensor(img))
    }

    for key, value in img_data.items():
        if key in TF_TENSORS:
            feature[key] = _bytes_feature(tf.io.serialize_tensor(value))
        else:
            feature[key] = _bytes_feature(value)


    example = tf.train.Example(features=tf.train.Features(feature=feature))

    writer.write(example.SerializeToString())
    
class SaveBuffer():
    def __init__(self, parent_dir, length, identifier):
        if parent_dir[-1] != "/":
            raise ValueError(f"expected parent directory {parent_dir} to end with /")
        
        if not isinstance(identifier, str):
            identifier = str(identifier)
            
        self.parent_dir = parent_dir
        self.length = length
        self.identifier = identifier
        self.buffer = []
        self.written_files = []

    def add(self, img, img_data):
        self.buffer.append((img, img_data))

        if len(self.buffer) >= self.length:
            self.__write()

    def __len__(self):
        return len(self.buffer)

    def is_empty(self):
        return len(self.buffer) == 0

    def flush(self):
        self.__write()
    
    def empty_out(self):
        tmp = self.buffer
        self.buffer = []

        return tmp

    def next_filename(self):
        return self.parent_dir + self.identifier + f"_{len(self.written_files)}.record"

    def __write(self):
        filename = self.next_filename()

        with tf.io.TFRecordWriter(filename) as writer:
            for img, img_data in self.buffer:
                write_img(img, img_data, writer)
        
        self.buffer = []
        self.written_files.append(filename)
