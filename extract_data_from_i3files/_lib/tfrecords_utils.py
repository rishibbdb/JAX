import tensorflow as tf
import numpy as np

def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))): # if value ist tensor
        value = value.numpy() # get value of tensor
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
  """Returns a floast_list from a float / double."""
  return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _int64_feature(value):
  """Returns an int64_list from a bool / enum / int / uint."""
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def serialize_array(array):
  array = tf.io.serialize_tensor(array)
  return array


def serialize_example(x, y):
    data = {
            "features": _bytes_feature(serialize_array(x)),
            "labels":  _bytes_feature(serialize_array(y)),
          }
    example = tf.train.Example(features=tf.train.Features(feature=data))
    return example.SerializeToString()
