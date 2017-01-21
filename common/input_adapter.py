#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: xiwen zhao
@created: 2016.11.14
"""

import data_manager
from util.indexer import Indexer


def get_label_indexer(key_subtask):
    if key_subtask == 'A':
        return Indexer(['negative', 'neutral', 'positive'])

    if key_subtask in ['B', 'D']:
        return Indexer(['negative', 'positive'])

    if key_subtask in ['C', 'E']:
        return Indexer(['-2', '-1', '0', '1', '2'])

    return None


def get_text_indexer(key_subtask):
    vocabs = data_manager.read_vocabs(key_subtask)

    return Indexer(vocabs)


def get_emo_label_indexer():
    f_emo_num = open('../data/clean/emo_nums_chosen_2.txt', 'r')
    lines = f_emo_num.readlines()
    emos= []

    for line in lines:
        emos.append(line.split('\t')[0])

    return Indexer(emos)


def get_emo_text_indexer():
    vocabs = data_manager.read_emo_vocabs()

    return Indexer(vocabs)
