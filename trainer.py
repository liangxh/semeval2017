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
        self.nb_epoch = 15
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

        self.test = test

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

        print 'Train...'
        self.model.fit(
            train[0], train[1],
            batch_size = self.batch_size,
            nb_epoch = self.nb_epoch,
            validation_data = valid
        )
        return self.model

    def evaluate(self):
        pred_classes = self.model.predict_classes([self.test[0], self.test[0]], batch_size = self.batch_size)
        tweet_id = []
        tweet_topic = []
        tweet_label = []
        if self.key_subtask == 'A':
            tweet_id_label = data_manager.read_id_label(self.key_subtask)
            for item in tweet_id_label:
                tweet_id.append(item[0])
                tweet_label.append(item[1])

            f_pred = open('pred_resultA.txt', 'w')

            for t_id, t_topic, result in zip(tweet_id, pred_classes):
                indexer = input_adapter.get_label_indexer(self.key_subtask)
                f_pred.write(t_id + '\t' + indexer.label(result) + '\n')
            f_pred.close()

            f_gold = open('gold_resultA.txt', 'w')

            for t_id, t_topic, t_label in zip(tweet_id, tweet_label):
                f_gold.write(t_id + '\t' + t_label + '\n')
            f_gold.close()

        elif self.key_subtask:
            tweet_id_topic_label = data_manager.read_id_topic_label(self.key_subtask)
            for item in tweet_id_topic_label:
                tweet_id.append(item[0])
                tweet_topic.append(item[1])
                tweet_label.append(item[2])

            f_pred = open('pred_result%s.txt'%(self.key_subtask), 'w')

            for t_id, t_topic, result in zip(tweet_id, tweet_topic, pred_classes):
                indexer = input_adapter.get_label_indexer(self.key_subtask)
                f_pred.write(t_id + '\t' + t_topic + '\t' + indexer.label(result) + '\n')
            f_pred.close()

            f_gold = open('gold_result%s.txt'%(self.key_subtask), 'w')

            for t_id, t_topic, t_label in zip(tweet_id, tweet_topic, tweet_label):
                f_gold.write(t_id + '\t' + t_topic + '\t' + t_label + '\n')
            f_gold.close()

        score, acc = self.model.evaluate([self.test[0], self.test[0]], self.test[1],
                           batch_size=self.batch_size)

        print('Test accuracy:', acc)


