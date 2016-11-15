# -*- coding: utf-8 -*-
"""
@author:
@created:
"""

# TODO(zxw) fill in the header and read the re-constructed code

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding
from keras.layers import Convolution1D, GlobalMaxPooling1D

def build(config, weights = {}):
    model = Sequential()

    # we start off with an efficient embedding layer which maps
    # our vocab indices into embedding_dims dimensions
    model.add(Embedding(config['max_features'],
                        config['embedding_dims'],
                        input_length = config['input_length'],
                        weights = [weights['Wemb']] if 'Wemb' in weights else None,
                        dropout = 0.2))

    # we add a Convolution1D, which will learn nb_filter
    # word group filters of size filter_length:
    model.add(Convolution1D(nb_filter = config['nb_filter'],
                            filter_length = config['filter_length'],
                            border_mode = 'valid',
                            activation = 'relu',
                            subsample_length = 1))

    # we use max pooling:
    model.add(GlobalMaxPooling1D())

    # We add a vanilla hidden layer:
    model.add(Dense(config['hidden_dims']))
    model.add(Dropout(0.2))
    model.add(Activation('relu'))

    # We project onto a single unit output layer, and squash it with a sigmoid:
    model.add(Dense(config['nb_classes']))
    model.add(Activation('sigmoid'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    return model
