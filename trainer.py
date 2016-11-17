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

from util import tokenizer
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
        self.text_indexer = input_adapter.get_text_indexer(self.key_subtask)
        self.label_indexer = input_adapter.get_label_indexer(self.key_subtask)
        

    def set_model_config(self, options):
        raise NotImplementedError


    def build_model(self, config, weights):
        raise NotImplementedError


    def prepare_X(self, texts):
        x = map(tokenizer.tokenize, texts)        
        x = map(self.text_indexer.idx, x)
        x = sequence.pad_sequences(x, maxlen = self.input_length)

        return self.post_prepare_X(x)


    def post_prepare_X(self, x):
        return x


    def prepare_Y(self, labels):
        y = self.label_indexer.idx(labels)
        y = np_utils.to_categorical(y)
        
        return y


    def prepare_XY(self, texts_labels):
        texts = map(lambda k:k[0], texts_labels)
        labels = map(lambda k:k[1], texts_labels)

        return self.prepare_X(texts), self.prepare_Y(labels)


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

        # TODO(zxw) save the output file somewhere else, how about a new folder in data/?
        #           try to implement this using data_manager

        test = self.prepare_XY(test)  

        pred_classes = self.model.predict_classes(test[0], batch_size = self.batch_size)

        tweet_id = []
        tweet_topic = []
        tweet_label = []

        if self.key_subtask == 'A':
            tweet_id_label = data_manager.read_id_label(self.key_subtask)
            for item in tweet_id_label:
                tweet_id.append(item[0])
                tweet_label.append(item[1])

            f_pred = open('pred_resultA.txt', 'w')

            # TODO(zxw) bug here before
            for t_id, result in zip(tweet_id, pred_classes):
                f_pred.write(t_id + '\t' + self.label_indexer.label(result) + '\n')
            f_pred.close()

            f_gold = open('gold_resultA.txt', 'w')

            # TODO(zxw) bug here before
            for t_id, t_label in zip(tweet_id, tweet_label):
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
                f_pred.write(t_id + '\t' + t_topic + '\t' + self.label_indexer.label(result) + '\n')
            f_pred.close()

            f_gold = open('gold_result%s.txt'%(self.key_subtask), 'w')

            for t_id, t_topic, t_label in zip(tweet_id, tweet_topic, tweet_label):
                f_gold.write(t_id + '\t' + t_topic + '\t' + t_label + '\n')
            f_gold.close()


        # TODO(zxw) this part can be skipped?
        score, acc = self.model.evaluate(test[0], test[1],
                           batch_size=self.batch_size)

        print 'Test accuracy:', acc
