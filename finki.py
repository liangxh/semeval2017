#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: xiwen zhao
@created: 2016.11.18
"""


from optparse import OptionParser
from trainer import BaseTrainer
from common import data_manager

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Embedding, Merge
from keras.layers import LSTM, SimpleRNN, GRU
from keras.layers import Embedding
from keras.layers import Convolution1D, GlobalMaxPooling1D
from keras.layers.wrappers import Bidirectional
from keras.optimizers import RMSprop, SGD


class Trainer(BaseTrainer):
    def post_prepare_X(self, x):
        return [x for i in range(2)]

    def set_model_config(self, options):
        self.config = dict(        
            nb_filter = options.nb_filter,
            filter_length = options.filter_length,
            hidden_dims = options.hidden_dims,
            # dropout_W = options.dropout_W,  # TODO: How to add dropout?
            # dropout_U = options.dropout_U,
        )

    def build_model(self, config, weights):
        # TODO(zxw) build model according to the paper
        # TODO(zxw) make the options controllable

        gru_model = Sequential()
        gru_model.add(Embedding(config['max_features'],
                                config['embedding_dims'],
                                input_length = config['input_length'],
                                weights = [weights['Wemb']] if 'Wemb' in weights else None))
                                #dropout = 0.2))
        gru_model.add(GRU(100, dropout_W=0.25, dropout_U=0.25))
        # gru_model.add(Dense(config['hidden_dims']))
        # gru_model.add(Activation('sigmoid'))

        blstm_model = Sequential()
        blstm_model.add(Embedding(config['max_features'],
                                  config['embedding_dims'],
                                  input_length = config['input_length'],
                                  weights = [weights['Wemb']] if 'Wemb' in weights else None))
        blstm_model.add(Bidirectional(LSTM(100, dropout_W=0.25, dropout_U=0.25)))

        cnn_model = Sequential()
        cnn_model.add(Embedding(config['max_features'],
                                config['embedding_dims'],
                                input_length = config['input_length'],
                                weights = [weights['Wemb']] if 'Wemb' in weights else None))
                                #dropout = 0.2))

        cnn_model.add(Convolution1D(nb_filter = config['nb_filter'],
                                    filter_length = config['filter_length'],
                                    border_mode = 'valid',
                                    activation = 'relu',
                                    subsample_length = 1))

        cnn_model.add(GlobalMaxPooling1D())
        # cnn_model.add(Dense(config['hidden_dims']))
        # cnn_model.add(Activation('sigmoid'))

        # merged model
        merged_model = Sequential()
        merged_model.add(Merge([gru_model, cnn_model], mode='concat', concat_axis=1))

        merged_model.add(Dropout(0.25))
        merged_model.add(Dense(config['nb_classes']))
        merged_model.add(Activation('softmax'))

        rmsprop = RMSprop(lr=0.001, rho=0.9, epsilon=1e-06)
        sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=False)
        merged_model.compile(loss='binary_crossentropy',
                             optimizer=rmsprop,
                             metrics=['accuracy'])

        return merged_model


def main():
    optparser = OptionParser()
    optparser.add_option("-t", "--task", dest = "key_subtask", default = "D")
    optparser.add_option("-e", "--embedding", dest = "fname_Wemb", default = "glove.twitter.27B.25d.txt")
    optparser.add_option("-d", "--hidden_dims", dest = "hidden_dims", type = "int", default = 250)
    optparser.add_option("-f", "--nb_filter", dest = "nb_filter", type = "int", default = 100)
    optparser.add_option("-l", "--filter_length", dest = "filter_length", type = "int", default = 3)
    # optparser.add_option("-w", "--dropout_W", dest = "dropout_W", type = "float", default = 0.25)
    # optparser.add_option("-u", "--dropout_U", dest = "dropout_U", type = "float", default = 0.25)
    opts, args = optparser.parse_args()

    trainer = Trainer(opts)
    trainer.train()

    test = data_manager.read_texts_labels(opts.key_subtask, 'devtest')
    trainer.evaluate(test)


if __name__ == '__main__':
    main() 

