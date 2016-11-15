# -*- coding: utf-8 -*-
"""
@author: xiwen zhao
@created: 2016.11.15
"""


from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding
from keras.layers import Convolution1D, GlobalMaxPooling1D
from keras.optimizers import SGD


def build(config, weights = {}):
    model = Sequential()

    model.add(Embedding(config['max_features'],
                        config['embedding_dims'],
                        input_length = config['input_length'],
                        weights = [weights['Wemb']] if 'Wemb' in weights else None,
                        dropout = 0.2))

    model.add(Convolution1D(nb_filter = config['nb_filter'],
                            filter_length = config['filter_length'],
                            border_mode = 'valid',
                            activation = 'relu',
                            subsample_length = 1))

    model.add(GlobalMaxPooling1D())

    model.add(Dense(config['hidden_dims']))
    model.add(Dropout(0.2))
    model.add(Activation('relu'))

    model.add(Dense(config['nb_classes']))
    #model.add(Dense(1))
    model.add(Activation('sigmoid'))

    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    return model
