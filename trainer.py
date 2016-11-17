#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: xiwen zhao
@created: 2016.11.15
"""

import numpy as np
np.random.seed(1337)  # for reproducibility

from keras.utils import np_utils
from keras.preprocessing import sequence

from common import data_manager, input_adapter, wordembed
from prepare_data import prepare_dataset, prepare_input

class BaseTrainer:
    def __init__(self, options):
        self.key_subtask = options.key_subtask
        self.fname_Wemb = options.fname_Wemb
        self.nb_epoch = 20
        self.batch_size = 32
        self.input_length = 45

        self.set_model_config(options)
        self.init_indexer()

    def init_indexer(self):
        vocabs = data_manager.read_vocabs(self.key_subtask)
        self.text_indexer = input_adapter.get_text_indexer(vocabs)
        self.label_indexer = input_adapter.get_label_indexer(self.key_subtask)
        

    def set_model_config(self, options):
        raise NotImplementedError


    def build_model(self, config, weights):
        raise NotImplementedError


    def prepare_XY(self, texts_labels):
        # turn raw texts and labels into indexes
        x, y = input_adapter.adapt_texts_labels(texts_labels, self.text_indexer, self.label_indexer)

        # turn indexes into data format specifically for keras model
        x = sequence.pad_sequences(x, maxlen = self.input_length)
        y = np_utils.to_categorical(y)
        return x, y

    
    def train(self):
        # load raw texts and labels
        train = data_manager.read_texts_labels(self.key_subtask, 'train')
        valid = data_manager.read_texts_labels(self.key_subtask, 'dev')

        nb_classes = len(set(map(lambda k:k[1], train)))

        # set weights for building model
        weights = dict(
            Wemb = wordembed.get(self.text_indexer.labels(), self.fname_Wemb),
        )
        
        # set parameters for building model according to dataset and weights
        self.config.update(dict(
            nb_classes = nb_classes,  # use set() to filter repetitive classes
            max_features = self.text_indexer.size(),
            input_length = self.input_length,
            embedding_dims = weights['Wemb'].shape[1],
        ))

        train = self.prepare_XY(train)    
        valid = self.prepare_XY(valid)

        self.model = self.build_model(self.config, weights)
       
        self.model.fit(
            *train,
            batch_size = self.batch_size,
            nb_epoch = self.nb_epoch,
            validation_data = valid
        )


    def evaluate(self, test):
        """
        Args
            test: a tuple of two lists: list of texts and list of labels
        """

        test = self.prepare_XY(test)  

        score, acc = self.model.evaluate(*test, batch_size = self.batch_size)
        print 'Test accuracy:', acc

