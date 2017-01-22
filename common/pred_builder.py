#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: xihao liang
@created: 2016.11.28 
"""

import os
import numpy as np
import data_manager
from util.indexer import Indexer


def build(key_subtask, mode, labels):
    return eval('build_%s' % key_subtask)(mode, labels)  # 返回build_A ~ build_E的结果


def build_A(mode, labels):
    key_subtask = 'A'
    rows = data_manager.read_data(key_subtask, mode)
    ofname = data_manager.fname_pred(key_subtask, mode)

    fobj = open(ofname, 'w')
    for row, label in zip(rows, labels):
        fobj.write('%s\t%s\n' % (row[0], label))
    fobj.close()


def build_B(mode, labels):
    key_subtask = 'B'
    rows = data_manager.read_data(key_subtask, mode)
    ofname = data_manager.fname_pred(key_subtask, mode)

    fobj = open(ofname, 'w')
    for row, label in zip(rows, labels):
        fobj.write('%s\t%s\t%s\n' % (row[0], row[1], label))
    fobj.close()


def build_C(mode, labels):
    key_subtask = 'C'
    rows = data_manager.read_data(key_subtask, mode)
    ofname = data_manager.fname_pred(key_subtask, mode)

    fobj = open(ofname, 'w')
    for row, label in zip(rows, labels):
        fobj.write('%s\t%s\t%s\n' % (row[0], row[1], label))
    fobj.close()


def build_D(mode, labels):
    key_subtask = 'D'
    rows = data_manager.read_data(key_subtask, mode)
    ofname = data_manager.fname_pred(key_subtask, mode)

    indexer = Indexer(['positive', 'negative'])

    topic_count = {}
    for row, label in zip(rows, labels):
        topic = row[1]

        if topic not in topic_count:
            topic_count[topic] = np.zeros(indexer.size())  # create a numpy array: [0., 0.]
        
        topic_count[topic][indexer.idx(label)] += 1

    fobj = open(ofname, 'w')
    for topic, count in topic_count.items():
        n_sample = np.sum(count)
        dist = count / n_sample

        fobj.write('%s\t%s\n' % (
                    topic,
                    '\t'.join(map(lambda k: '%.12f' % k, dist)),
                    ))

    fobj.close()


def build_E(mode, labels):
    key_subtask = 'E'
    rows = data_manager.read_data(key_subtask, mode)
    ofname = data_manager.fname_pred(key_subtask, mode)

    indexer = Indexer(['-2', '-1', '0', '1', '2'])

    topic_count = {}
    for row, label in zip(rows, labels):
        topic = row[1]

        if topic not in topic_count:
            topic_count[topic] = np.zeros(indexer.size())
        
        topic_count[topic][indexer.idx(label)] += 1

    fobj = open(ofname, 'w')
    for topic, count in topic_count.items():
        n_sample = np.sum(count)
        dist = count / n_sample
        fobj.write('%s\t%s\n'%(
                    topic,
                    '\t'.join(map(lambda k: '%.12f' % k, dist))
                    ))

    fobj.close()
