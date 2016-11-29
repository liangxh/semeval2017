#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: xiwen zhao
@created: 2016.11.15
"""

import global_config

import commands
import numpy as np

if hasattr(global_config, 'NP_RANDOM_SEED'):
    np.random.seed(global_config.NP_RANDOM_SEED)  # for reproducibility

from keras.utils import np_utils
from keras.preprocessing import sequence
from keras.callbacks import ModelCheckpoint, Callback

from util import tokenizer
from common import data_manager, input_adapter, wordembed, pred_builder, gold_builder


class BaseTrainer:
    def __init__(self, options):
        self.key_subtask = options.key_subtask
        self.fname_Wemb = options.fname_Wemb
        self.nb_epoch = global_config.nb_epoch
        self.batch_size = global_config.batch_size
        self.input_length = global_config.input_length
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
        x = sequence.pad_sequences(x, maxlen=self.input_length)

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

    def save_model_weight(self):
        fname = data_manager.fname_model_weight(self.key_subtask, self.model_name)
        self.model.save_weights(fname)

    def load_model_weight(self):
        fname = data_manager.fname_model_weight(self.key_subtask, self.model_name)
        self.model.load_weights(fname)

    def train(self):
        # load raw texts and labels
        train = data_manager.read_texts_labels(self.key_subtask, 'train')
        valid = data_manager.read_texts_labels(self.key_subtask, 'dev')

        nb_classes = len(set(map(lambda k:k[1], train)))  # use set() to filter repetitive classes

        # set weights for building model
        weights = dict(
            Wemb=wordembed.get(self.text_indexer.labels(), self.fname_Wemb),
        )

        # set parameters for building model according to dataset and weights
        self.config.update(dict(
            nb_classes = nb_classes,
            max_features = self.text_indexer.size(),
            input_length = self.input_length,
            embedding_dims = weights['Wemb'].shape[1],
        ))

        train = self.prepare_XY(train)
        valid = self.prepare_XY(valid)

        self.model = self.build_model(self.config, weights)
        self.save_model_config()

        '''
        checkpoint = ModelCheckpoint(
                                 data_manager.fname_model_weight(self.key_subtask, self.model_name),
                                 monitor = 'val_acc',
                                 verbose = 0,
                                 save_best_only = True,
                                 save_weights_only = True,
                                 mode = 'auto'
                                )
        '''

        bestscore = SaveBestScore(self)

        self.model.fit(
            train[0], train[1],
            batch_size=self.batch_size,
            nb_epoch=self.nb_epoch,
            validation_data=valid,
            callbacks=[bestscore, ]
        )

    def pred_classes(self, texts):
        X = self.prepare_X(texts)
        Y = self.model.predict_classes(X, batch_size=self.batch_size)
        labels = map(self.label_indexer.label, Y)

        return labels
    
    def evaluate(self, mode='devtest'):
        texts = data_manager.read_texts(self.key_subtask, mode)
        labels = self.pred_classes(texts)
        pred_builder.build(self.key_subtask, mode, labels)

        o = commands.getoutput(
             "perl eval/score-semeval2016-task4-subtask%s.pl " \
             "../data/result/%s_%s_gold.txt " \
             "../data/result/%s_%s_pred.txt" % (
                self.key_subtask, 
                self.key_subtask, mode, 
                self.key_subtask, mode,
            )
        )

        try:
            o = o.strip()
            lines = o.split("\n")
            score = lines[-1].split("\t")[-1]
            return float(score)
        except:
            print "trainer.evaluate: [warning] invalid output file for semeval measures tool"
            print [o, ]
            return None

    def simple_evaluate(self, test):
        test = self.prepare_XY(test)
        return self.model.evaluate(*test, batch_size=self.batch_size)


class SaveBestScore(Callback):
    def __init__(self, trainer):
        self.trainer = trainer
        self.key_subtask = trainer.key_subtask
        self.score = 0
        self.best_score = None
        self.max_valacc = 0

        super(Callback, self).__init__()

    def on_epoch_end(self, epoch, logs={}):
        self.score = self.trainer.evaluate('dev')
        print ' - val_score:', self.score

        if logs.get('val_acc') > self.max_valacc:
            self.max_valacc = logs.get('val_acc')

        if self.best_score is None:
            self.best_score = self.score

        if self.key_subtask == 'A' or self.key_subtask == 'B':
            if self.score > self.best_score:
                self.best_score = self.score
                self.trainer.save_model_weight()

        if self.score < self.best_score:
            self.best_score = self.score
            self.trainer.save_model_weight()

    def on_train_end(self, logs={}):
        print 'maximum val_acc: ', self.max_valacc
        print 'best score:', self.best_score


