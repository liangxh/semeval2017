#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author:
@created:
"""

# TODO(zxw) fill in the header and read the re-constructed code

from keras.utils import np_utils
from keras.preprocessing import sequence

from common import data_manager, input_adapter, wordembed

import numpy as np
np.random.seed(1337)  # for reproducibility

def prepare_dataset(key_subtask, vocabs):
    text_indexer = input_adapter.get_text_indexer(vocabs)
    label_indexer = input_adapter.get_label_indexer(key_subtask)

    def prepare_dataset(mode):
        texts_labels = data_manager.read_texts_labels(key_subtask, mode)
        x, y = input_adapter.adapt_texts_labels(texts_labels, text_indexer, label_indexer)
        return x, y

    dataset = tuple([prepare_dataset(mode) for mode in ['train', 'dev', 'devtest']])

    return dataset

def prepare_input(xy, input_length):
    x, y = xy
    x = sequence.pad_sequences(x, maxlen = input_length)
    y = np_utils.to_categorical(y)
    return x, y

# set parameters for input
key_subtask = 'A'
wemb_file = 'glove.twitter.27B.25d.txt'

# set parameters for training
nb_epoch = 20
batch_size = 32
input_length = 45

# load data from files
print 'Loading data...'

vocabs = data_manager.read_vocabs(key_subtask)
dataset = prepare_dataset(key_subtask, vocabs)
train, valid, test = map(lambda dset: prepare_input(dset, input_length), dataset)

# set weights for building model
weights = dict(
    Wemb = wordembed.get(vocabs, wemb_file),
)

# set parameters for building model according to dataset and weights
config = dict(
    # parameters related to the dataset
    nb_classes = len(set(dataset[0][1])),
    max_features = len(vocabs),
    input_length = input_length,
    embedding_dims = weights['Wemb'].shape[1],

    # CNN
    nb_filter = 250,
    filter_length = 3,
    hidden_dims = 250,
)

print 'Build model...'
from model import cnn as Model
model = Model.build(config, weights)

model.fit(
    train[0], train[1],
    batch_size = batch_size,
    nb_epoch = nb_epoch,
    validation_data = valid
)

score, acc = model.evaluate(
                test[0], test[1],
                batch_size = batch_size,
            )

print 'Test accuracy:', acc
