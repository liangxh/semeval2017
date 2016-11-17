#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: xiwen zhao
@created: 2016.11.15
"""

import numpy as np
np.random.seed(1337)  # for reproducibility

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


    def set_model_config(self, options):
        raise NotImplementedError


    def build_model(self, config, weights):
        raise NotImplementedError


    def train(self):
        # load data from files
        print 'Loading data...'
        vocabs = data_manager.read_vocabs(self.key_subtask)
        dataset = prepare_dataset(self.key_subtask, vocabs)
        train, valid, test = map(
                    lambda dset: prepare_input(dset, self.input_length),
                    dataset)

        # set weights for building model
        weights = dict(
            Wemb = wordembed.get(vocabs, self.fname_Wemb),
        )
        
        # set parameters for building model according to dataset and weights
        self.config.update(dict(
            nb_classes = len(set(dataset[0][1])),  # use set() to filter repetitive classes
            max_features = len(vocabs),
            input_length = self.input_length,
            embedding_dims = weights['Wemb'].shape[1],
        ))
        
        self.model = self.build_model(self.config, weights)

        self.model.fit(
            *train,
            batch_size = self.batch_size,
            nb_epoch = self.nb_epoch,
            validation_data = valid
        )

        score, acc = self.model.evaluate(*test, batch_size = self.batch_size)

        print 'Test accuracy:', acc

