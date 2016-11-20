#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: xiwen zhao
@created: 2016.11.15
"""

import numpy as np
#np.random.seed(1337)  # for reproducibility

from keras.utils import np_utils
from keras.preprocessing import sequence
from keras.callbacks import ModelCheckpoint

from util import tokenizer
from common import data_manager, input_adapter, wordembed


class BaseTrainer:
    def __init__(self, options):
        self.key_subtask = options.key_subtask
        self.fname_Wemb = options.fname_Wemb
        self.nb_epoch = 15
        self.batch_size = 32
        self.input_length = 45

        self.set_model_config(options)
        self.init_indexer()
        self.model_name = self.get_model_name()


    def get_model_name(self):
        """
        return __file__.split('/')[-1].split('.')[0]
        """
        raise NotImplementedError


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


    def save_model_config(self):
        fname = data_manager.fname_model_config(self.key_subtask, self.model_name)
        open(fname, 'w').write(self.model.to_json())


    def load_model_weight(self):
        fname = data_manager.fname_model_weight(self.key_subtask, self.model_name)
        self.model.load_weights(fname)


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
        self.save_model_config()        

        checkpoint = ModelCheckpoint(
                        data_manager.fname_model_weight(self.key_subtask, self.model_name),
                        monitor = 'val_acc',
                        verbose = 0,
                        save_best_only = True,
                        save_weights_only = True,
                        mode = 'max'
                    )

        self.model.fit(
            *train,
            batch_size = self.batch_size,
            nb_epoch = self.nb_epoch,
            validation_data = valid,
            callbacks = [checkpoint, ]
        )


    def evaluate(self, test):
        """
        Args
            test: a tuple of two lists: list of texts and list of labels
        """

        test = self.prepare_XY(test)

        pred_classes = self.model.predict_classes(test[0], batch_size = self.batch_size)

        if self.key_subtask == 'A':
            data_manager.write_id_label(self.key_subtask, pred_classes)

        elif self.key_subtask == 'B' or self.key_subtask == 'C':
            data_manager.write_id_topic_label(self.key_subtask, pred_classes)
        elif self.key_subtask == 'D':
            data_manager.write_topic_label(self.key_subtask, pred_classes)
        else:
            data_manager.write_topic_5labels(self.key_subtask, pred_classes)


    def simple_evaluate(self, test):
        test = self.prepare_XY(test)
        return self.model.evaluate(*test, batch_size = self.batch_size)

