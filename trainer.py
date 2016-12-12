#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: xiwen zhao
@created: 2016.11.15
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


class BaseTrainer:
    def __init__(self, options):
        self.key_subtask = options.key_subtask
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
        self.text_indexer = input_adapter.get_text_indexer(self.key_subtask)
        self.label_indexer = input_adapter.get_label_indexer(self.key_subtask)

    def set_model_config(self, options):
        raise NotImplementedError

    def build_model(self, config, weights):
        raise NotImplementedError

    def build_pre_model(self, config, weights):
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

        if self.config['nb_classes'] > 2:
            y = np_utils.to_categorical(y)

        return y

    def prepare_XY(self, texts_labels):
        texts = map(lambda k:k[0], texts_labels)
        labels = map(lambda k:k[1], texts_labels)

        return self.prepare_X(texts), self.prepare_Y(labels)

    def prepare_XY_emo(self, texts_labels):
        texts = map(lambda k:k[0], texts_labels)
        labels = map(lambda k:k[1], texts_labels)

        return self.prepare_X(texts), labels

    def save_model_config(self):
        fname = data_manager.fname_model_config(self.key_subtask, self.model_name)
        open(fname, 'w').write(self.model.to_json())

    def save_model_weight(self):
        fname = data_manager.fname_model_weight(self.key_subtask, self.model_name)
        self.model.save_weights(fname)

    def load_model_weight(self):
        fname = data_manager.fname_model_weight(self.key_subtask, self.model_name)
        self.model.load_weights(fname)

    def train(self, weights):
        # load raw texts and labels
        # train = data_manager.read_texts_labels(self.key_subtask, 'train')
        # dev = data_manager.read_texts_labels(self.key_subtask, 'dev')

        train = data_manager.read_texts_labels(self.key_subtask, 'train_dev')
        dev = data_manager.read_texts_labels(self.key_subtask, 'devtest')

        nb_classes = len(set(map(lambda k:k[1], train)))  # use set() to filter repetitive classes

        # set weights for building model
        '''
        weights = dict(
            Wemb=wordembed.get(self.text_indexer.labels(), self.fname_Wemb),
        )
        '''
        # set parameters for building model according to dataset and weights
        self.config.update(dict(
            nb_classes = nb_classes,
            max_features = self.text_indexer.size(),
            input_length = self.input_length,
            embedding_dims = weights['Wemb'].shape[1],
        ))

        train = self.prepare_XY(train)
        dev = self.prepare_XY(dev)

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
            validation_data=dev,
            callbacks=[bestscore, ]
        )

        bestscore.export_history()

    def pre_train(self):
        train = data_manager.read_emo_texts_labels(self.key_subtask, 'train')
        dev = data_manager.read_emo_texts_labels(self.key_subtask, 'dev')

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

        train = self.prepare_XY_emo(train)
        dev = self.prepare_XY_emo(dev)

        self.model = self.build_pre_model(self.config, weights)
        # self.save_model_config()

        self.model.fit(
            train[0], train[1],
            batch_size=self.batch_size,
            nb_epoch=self.nb_epoch,
            validation_data=dev,
        )

        return self.save_model_weight()

    def pred_prob(self):
        # train = data_manager.read_texts_labels(self.key_subtask, 'train')
        # dev = data_manager.read_texts_labels(self.key_subtask, 'dev')
        # devtest = data_manager.read_texts_labels(self.key_subtask, 'devtest')

        train = data_manager.read_texts_labels(self.key_subtask, 'train_dev')
        dev = data_manager.read_texts_labels(self.key_subtask, 'devtest')
        test = data_manager.read_texts_labels(self.key_subtask, 'test_new')

        weights = dict(
            Wemb=wordembed.get(self.text_indexer.labels(), self.fname_Wemb),
        )

        nb_classes = len(set(map(lambda k:k[1], train)))
        self.config.update(dict(
            nb_classes = nb_classes,
        ))

        self.model = self.build_model(self.config, weights)
        fname = '../data/model/subtask%s_%s_weight_new.hdf5' % (self.key_subtask, self.config['model_name'])
        self.model.load_weights(fname)
        # self.load_model_weight()

        train = self.prepare_XY(train)
        dev = self.prepare_XY(dev)
        # devtest = self.prepare_XY(devtest)
        test = self.prepare_XY(test)

        # for name, data in zip(['train', 'dev', 'devtest', 'test_new'], [train, dev, devtest, test]):
        for name, data in zip(['train', 'dev', 'test_new'], [train, dev, test]):
            results = self.model.predict_proba(data[0], batch_size=self.batch_size)
            topics = data_manager.read_topic(self.key_subtask, name)

            fname = '../data/pred_prob/%s_%s_%s_new.txt' % (self.key_subtask, name, self.config['model_name'])
            with open(fname, 'w') as f:
                if topics is not None:
                    for result, topic in zip(results, topics):
                        f.write(topic + '\t' + '\t'.join(map(str, result)) + '\n')
                else:
                    for result in results:
                        f.write('\t'.join(map(str, result)) + '\n')

    def pred_classes(self, texts, verbose=0):
        X = self.prepare_X(texts)
        Y = self.model.predict_classes(X, batch_size=self.batch_size, verbose=verbose)
        labels = map(self.label_indexer.label, Y)

        return labels
    
    def evaluate(self, mode='test_new', verbose=0):
        texts = data_manager.read_texts(self.key_subtask, mode)
        labels = self.pred_classes(texts, verbose)
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
        self.prior_score = (lambda a, b: a > b) \
            if self.key_subtask in ['A', 'B'] else (lambda a, b: a < b)
        self.dev_scores = []
        self.devtest_scores = []
        self.num_epoch = 0
        self.best_epoch = 0

        super(Callback, self).__init__()

    def on_epoch_end(self, epoch, logs={}):
        self.num_epoch += 1

        # self.score = self.trainer.evaluate('dev')
        self.score = self.trainer.evaluate('devtest')
        # print ' - val_score: %f' % self.score

        # devtest_score = self.trainer.evaluate('devtest')
        devtest_score = self.trainer.evaluate('test_new')
        self.dev_scores.append(self.score)
        self.devtest_scores.append(devtest_score)
        # print ' - devtest_score: %f'%(devtest_score)

        if logs.get('val_acc') > self.max_valacc:
            self.max_valacc = logs.get('val_acc')

        if self.best_score is None or self.prior_score(self.score, self.best_score):
            self.best_score = self.score
            self.best_epoch = self.num_epoch
            self.trainer.save_model_weight()

    def on_train_end(self, logs={}):
        print 'maximum val_acc: ', self.max_valacc
        print 'best score:', self.best_score, ' corresponding epoch number:', self.best_epoch

    def export_history(self):
        fname = os.path.join(data_manager.DIR_RESULT, '%s_history_new.json' % self.trainer.model_name)
        cPickle.dump((self.dev_scores, self.devtest_scores), open(fname, 'w'))

