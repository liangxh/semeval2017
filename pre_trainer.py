#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: xiwen zhao
@created: 2016.12.13
"""

import global_config

import os
import cPickle
import commands
import numpy as np

if hasattr(global_config, 'NP_RANDOM_SEED'):
    np.random.seed(global_config.NP_RANDOM_SEED)  # for reproducibility

from keras.utils import np_utils
from keras.preprocessing import sequence
from keras.callbacks import ModelCheckpoint, Callback

from util import tokenizer
from common import data_manager, input_adapter, wordembed, pred_builder


class BasePreTrainer:
    def __init__(self, options):
        self.fname_Wemb = options.fname_Wemb
        self.nb_epoch = options.nb_epoch
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
        self.text_indexer = input_adapter.get_emo_text_indexer()
        self.label_indexer = input_adapter.get_emo_label_indexer()

    def set_model_config(self, options):
        raise NotImplementedError

    def build_pre_model(self, config, weights):
        raise NotImplementedError

    def post_prepare_X(self, x):
        return x

    def prepare_X_emo(self, texts):
        x = map(tokenizer.tokenize, texts)
        x = map(self.text_indexer.idx, x)
        x = sequence.pad_sequences(x, maxlen=self.input_length)

        return self.post_prepare_X(x)

    def prepare_Y_emo(self, labels):
        y = self.label_indexer.idx(labels)

        # if self.config['nb_classes'] > 2:
        y = np_utils.to_categorical(y, self.config['nb_classes'])

        return y

    def prepare_XY_emo(self, texts_labels):
        texts = map(lambda k:k[0], texts_labels)
        labels = map(lambda k:k[1], texts_labels)

        return self.prepare_X_emo(texts), self.prepare_Y_emo(labels)

    def save_pretrain_model_config(self):
        fname = data_manager.fname_pretrain_model_config(self.model_name)
        open(fname, 'w').write(self.model.to_json())

    def save_pretrain_model_weight(self):
        print '\nSaving pretrain model weight for %s...' % self.model_name
        fname = data_manager.fname_pretrain_model_weight(self.model_name)
        self.model.save_weights(fname)

    def pre_train(self):
        train = data_manager.read_emo_texts_labels('train_cut_new')
        dev = data_manager.read_emo_texts_labels('dev_cut_new')

        emos = open('../data/clean/emo_nums_chosen_2.txt', 'r').readlines()
        nb_classes = len(emos)
        print 'nb_classes:', nb_classes

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

        train = self.prepare_XY_emo(train)
        dev = self.prepare_XY_emo(dev)
      
        self.model = self.build_pre_model(self.config, weights)
        self.save_pretrain_model_config()

        bestscore = SaveBestScore(self)

        self.model.fit(
            train[0], train[1],
            batch_size=self.batch_size,
            nb_epoch=self.nb_epoch,
            validation_data=dev,
            callbacks=[bestscore, ]
        )

        bestscore.export_history()

    def pred_classes(self, texts, verbose=0):
        X = self.prepare_X_emo(texts)
        Y = self.model.predict_classes(X, batch_size=self.batch_size, verbose=verbose)
        labels = map(self.label_indexer.label, Y)

        return labels
    """
    def simple_evaluate(self, mode='dev_cut'):
        dev = data_manager.read_emo_texts_labels(mode)
        dev = self.prepare_XY_emo(dev)
        return self.model.evaluate(*dev, batch_size=self.batch_size)
    """
    def export_pred(self, mode="dev_cut_new"):
        texts = data_manager.read_emo_texts(mode)
        labels = self.pred_classes(texts)

        ofname = data_manager.fname_pretrain_pred(self.model_name)
        fobj = open(ofname, 'w')

        for text, label in zip(texts, labels):
            fobj.write('%s\t%s\n' % (label, text))

        fobj.close()


class SaveBestScore(Callback):
    def __init__(self, trainer):
        self.trainer = trainer
        self.score = 0
        self.best_score = None
        self.max_valacc = 0
        self.num_epoch = 0
        self.best_epoch = 0
        self.accs = []
        self.val_accs =[]

        super(Callback, self).__init__()

    def on_epoch_end(self, epoch, logs={}):
        self.num_epoch += 1

        self.accs.append(logs.get('acc'))
        self.val_accs.append(logs.get('val_acc'))

        if logs.get('val_acc') > self.max_valacc:
            self.max_valacc = logs.get('val_acc')
            self.best_epoch = self.num_epoch
            self.trainer.save_pretrain_model_weight()

    def on_train_end(self, logs={}):
        print 'maximum val_acc: ', self.max_valacc

    def export_history(self):
        fname = os.path.join(data_manager.DIR_HISTORY, '%s_history.json' % self.trainer.model_name)
        cPickle.dump((self.accs, self.val_accs), open(fname, 'w'))
