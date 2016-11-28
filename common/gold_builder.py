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


def build(key_subtask, mode):
    return eval('build_%s'%(key_subtask))(mode)


def build_A(mode):
    key_subtask = 'A'
    rows = data_manager.read_data(key_subtask, mode)
    ofname = data_manager.fname_gold(key_subtask, mode)

    fobj = open(ofname, 'w')
    for row in rows:
        fobj.write('%s\t%s\n'%(row[0], row[1]))
    fobj.close()


def build_B(mode):
    key_subtask = 'B'
    rows = data_manager.read_data(key_subtask, mode)
    ofname = data_manager.fname_gold(key_subtask, mode)

    fobj = open(ofname, 'w')
    for row in rows:
        fobj.write('%s\t%s\t%s\n'%(row[0], row[1], row[2]))
    fobj.close()


def build_C(mode):
    key_subtask = 'C'
    rows = data_manager.read_data(key_subtask, mode)
    ofname = data_manager.fname_gold(key_subtask, mode)

    fobj = open(ofname, 'w')
    for row in rows:
        fobj.write('%s\t%s\t%s\n'%(row[0], row[1], row[2]))
    fobj.close()


def build_D(mode):
    key_subtask = 'D'
    rows = data_manager.read_data(key_subtask, mode)
    ofname = data_manager.fname_gold(key_subtask, mode)

    indexer = Indexer(['positive', 'negative'])

    topic_label = {}
    for row in rows:
        topic = row[1]
        label = row[2]

        if not topic in topic_label:
            topic_label[topic] = np.zeros(indexer.size())
        
        topic_label[topic][indexer.idx(label)] += 1

    fobj = open(ofname, 'w')
    for topic, labels in topic_label.items():
        n_sample = np.sum(labels)
        dist = labels / n_sample
        fobj.write('%s\t%s\t%d\n'%(
                    topic,
                    '\t'.join(map(lambda k: '%.10f'%(k), dist)),
                    int(n_sample)
                    ))

    fobj.close()


def build_E(mode):
    key_subtask = 'E'
    rows = data_manager.read_data(key_subtask, mode)
    ofname = data_manager.fname_gold(key_subtask, mode)

    indexer = Indexer(['-2', '-1', '0', '1', '2'])

    topic_label = {}
    for row in rows:
        topic = row[1]
        label = row[2]

        if not topic in topic_label:
            topic_label[topic] = np.zeros(indexer.size())
        
        topic_label[topic][indexer.idx(label)] += 1

    fobj = open(ofname, 'w')
    for topic, labels in topic_label.items():
        n_sample = np.sum(labels)
        dist = labels / n_sample
        fobj.write('%s\t%s\n'%(
                    topic,
                    '\t'.join(map(lambda k: '%.10f'%(k), dist))                    ))

    fobj.close()
