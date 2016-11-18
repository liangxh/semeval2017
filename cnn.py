#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: xiwen zhao
@created: 2016.11.15
"""

from optparse import OptionParser
from trainer import BaseTrainer
from common import data_manager

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding
from keras.layers import Convolution1D, GlobalMaxPooling1D
from keras.optimizers import SGD

class Trainer(BaseTrainer):
    def set_model_config(self, options):
        self.config = dict(        
            nb_filter = options.nb_filter,
            filter_length = options.filter_length,
            hidden_dims = options.hidden_dims,
        )

    def build_model(self, config, weights):
        print 'Build cnn model...'
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
        model.add(Activation('sigmoid'))

        sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=False)
        model.compile(loss='binary_crossentropy',
                      optimizer=sgd,
                      metrics=['accuracy'])

        return model


def main():
    optparser = OptionParser()
    optparser.add_option("-t", "--task", dest = "key_subtask", default = "C")
    optparser.add_option("-e", "--embedding", dest = "fname_Wemb", default = "glove.twitter.27B.25d.txt")
    optparser.add_option("-d", "--hidden_dims", dest = "hidden_dims", type = "int", default = 250)
    optparser.add_option("-f", "--nb_filter", dest = "nb_filter", type = "int", default = 250)
    optparser.add_option("-l", "--filter_length", dest = "filter_length", type = "int", default = 3)
    opts, args = optparser.parse_args()

    trainer = Trainer(opts)
    model = trainer.train()
    return model

    test = data_manager.read_texts_labels(opts.key_subtask, 'devtest')
    trainer.evaluate(test)


if __name__ == '__main__':
    main() 

