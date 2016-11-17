#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: xiwen zhao
@created: 2016.11.17
"""

from optparse import OptionParser
from trainer import BaseTrainer

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding
from keras.layers import GRU
from keras.optimizers import SGD

class Trainer(BaseTrainer):
    def set_model_config(self, options):
        self.config = dict(
            nb_filter = options.nb_filter,
            filter_length = options.filter_length,
            hidden_dims = options.hidden_dims,
        )

    def build_model(self, config, weights):
        print 'Build grnn model...'
        model = Sequential()
        model.add(Embedding(config['max_features'],
                                config['embedding_dims'],
                                input_length = config['input_length'],
                                weights = [weights['Wemb']] if 'Wemb' in weights else None,
                                dropout = 0.2))
        model.add(GRU(128, dropout_W=0.2, dropout_U=0.2))

        # model.add(Dense(1))
        model.add(Dense(config['nb_classes']))
        model.add(Activation('sigmoid'))

        model.compile(loss='binary_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])

        return model


def main():
    optparser = OptionParser()
    optparser.add_option("-t", "--task", dest = "key_subtask", default = "B")
    optparser.add_option("-e", "--embedding", dest = "fname_Wemb", default = "glove.twitter.27B.25d.txt")
    optparser.add_option("-d", "--hidden_dims", dest = "hidden_dims", type = "int", default = 250)
    optparser.add_option("-f", "--nb_filter", dest = "nb_filter", type = "int", default = 250)
    optparser.add_option("-l", "--filter_length", dest = "filter_length", type = "int", default = 3)
    opts, args = optparser.parse_args()

    trainer = Trainer(opts)
    model = trainer.train()
    return model

if __name__ == '__main__':
    main()

