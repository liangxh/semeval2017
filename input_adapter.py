#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: xiwen zhao
@created: 2016.11.14
"""


import tokenizer
import data_manager
from indexer import Indexer

def get_label_indexer(key_subtask):
    if key_subtask == 'A':
        return Indexer(['negative', 'neutral', 'positive'])

    if key_subtask in ['B', 'D']:
        return Indexer(['negative', 'positive'])

    if key_subtask in ['C', 'E']:
        return Indexer(['-2', '-1', '0', '1', '2'])

    return None


def get_text_indexer(vocabs):
    return Indexer(vocabs)


def adapt_texts_labels(texts_labels, text_indexer, label_indexer):
    """
    Used for model training

    Args:
        texts_labels: list of tuple, for example: [('how are you', 'positive'), ...]
        text_indexer: an indexer for text; obtained by $get_text_indexer
        label_indexer: an indexer for label; obtained by $get_label_indexer    

    Return:
        two lists; the first list is composed of list of lists of tokenIDs
        the second list is composed of list of indexed labels
    """

    x = []
    y = []
    for text, label in texts_labels:
        tokens = tokenizer.tokenize(text)

        x.append(text_indexer.idx(tokens))
        y.append(label_indexer.idx(label))

    return x, y


def test():
    key_subtask = 'A'
    mode = 'train'  # CAUTION: input has no labels

    #vocabs = data_manager.read_vocab_minC(key_subtask, 1)
    vocabs = data_manager.read_vocab_topN(key_subtask, 4000)

    dataset = []
    for mode in ['train', 'dev', 'devtest']:
        texts_labels = data_manager.read_texts_labels(key_subtask, mode)

        text_indexer = get_text_indexer(vocabs)
        label_indexer = get_label_indexer(key_subtask)


        x, y = adapt_texts_labels(texts_labels, text_indexer, label_indexer)
        dataset.append((x, y))

    dataset = tuple(dataset)  # list of 3 tuples --> tuple of tuple
    train, dev, devtest = dataset
    train_X, train_Y = train     
    print train_X[0]
    print train_Y[0]  


if __name__ == '__main__':
    test()
