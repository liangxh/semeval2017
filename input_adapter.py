#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: 
@created:
"""

# TODO(zxw) fill in the header

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
    # TODO(zxw) implement this function like adapt_x; HINT: use get_indexer

    y = []
    return y 


def test():
    key_subtask = 'A'
    mode = 'train'

    # TODO(zxw) initialize vocabs using data_manager.read_vocab_XXXX
    vocabs = []

    # TODO(zxw) read texts (list of string) and labels (list of string) using data_manager.read_texts_labels
    texts = []
    labels = []

    x = adapt_x(texts, vocabs)
    y = adapt_y(labels, key_subtask)

    print x[1] # list of tokenID is expected
    print y[1] # an integer is expected

if __name__ == '__main__':
    test()
