#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: xiwen zhao
@created: 2016.11.15
"""
from common import data_manager, input_adapter, wordembed
from prepare_data import prepare_dataset, prepare_input

import numpy as np
# np.random.seed(1337)  # for reproducibility

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
# weights['Wemb'].shape == (5635, 25)

# set parameters for building model according to dataset and weights
config = dict(
    # parameters related to the dataset
    nb_classes = len(set(dataset[0][1])),  # use set() to filter repetitive classes
    max_features = len(vocabs),
    input_length = input_length,
    embedding_dims = weights['Wemb'].shape[1],

    # CNN
    nb_filter = 250,
    filter_length = 3,
    hidden_dims = 250,
)

print 'Build CNN model...'
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
'''
loss='binary_crossentropy' optimizer = 'adam'  0.71358
loss='binary_crossentropy' optimizer = 'sgd'  0.72955
loss='binary_crossentropy' optimizer = sgd  0.74354 -- lr=0.01, decay=1e-6, momentum=0.9, nesterov=False
'''
