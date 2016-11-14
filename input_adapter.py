#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: xiwen zhao
@created: 2016.11.14
"""


import tokenizer
import data_manager
from indexer import Indexer

def get_indexer(key_subtask):
    if key_subtask == 'A':
        return Indexer(['negative', 'neutral', 'positive'])

    if key_subtask in ['B', 'D']:
        return Indexer(['negative', 'positive'])

    if key_subtask in ['C', 'E']:
        return Indexer(['-2', '-1', '0', '1', '2'])

    return None


def adapt_x(texts, vocabs):
    indexer = Indexer(vocabs)

    texts = map(tokenizer.tokenize, texts)
    x = map(indexer.idx, texts)

    return x


def adapt_y(labels, key_subtask):
    indexer = get_indexer(key_subtask)
    y = map(indexer.idx, labels)

    return y 


def test():
    key_subtask = 'A'
    mode = 'train'  # CAUTION: input has no labels

    #vocabs = data_manager.read_vocab_minC(key_subtask, 1)
    vocabs = data_manager.read_vocab_topN(key_subtask, 4000)

    labels_texts = data_manager.read_texts_labels(key_subtask, mode)
    labels = map(lambda k: k[0], labels_texts)
    texts = map(lambda k: k[1], labels_texts)

    x = adapt_x(texts, vocabs)
    y = adapt_y(labels, key_subtask)

    print x[1] # list of tokenID is expected
    print y[1] # an integer is expected

if __name__ == '__main__':
    test()
