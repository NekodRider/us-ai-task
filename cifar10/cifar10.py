import numpy as np
import pandas as pd
import tensorflow as tf


def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dic = pickle.load(fo, encoding='bytes')
    return dic


def get_data():
    data_batch = []
    test_batch = unpickle('./cifar-10-batches-py/test_batch')
    labels = [i.decode() for i in unpickle('./cifar-10-batches-py/batches.meta')[b'label_names']]
    for i in range(1, 6):
        data_batch.append(unpickle('./cifar-10-batches-py/data_batch_' + str(i)))
    return data_batch, test_batch, labels


dataBatch, testBatch, labelList = get_data()

print(labelList)
