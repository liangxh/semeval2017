#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: xiwen zhao
@created: 2016.11.13
"""

import os

DIR_DATA = os.path.join(os.path.dirname(__file__), '../data')
DIR_RAW = os.path.join(DIR_DATA, 'raw')
DIR_CLEAN = os.path.join(DIR_DATA, 'clean')
DIR_WORDCOUNT = os.path.join(DIR_DATA, 'wordcount')

def unify_subtask_key(key):
    key = key.upper()

    if key in ["B", "D"]: return "BD"
    if key in ["C", "E"]: return "CE"
    
    return key


def fname_raw(key_subtask, mode):
    key = unify_subtask_key(key_subtask)
    return os.path.join(DIR_RAW, 'subtask%s_%s.tsv'%(key, mode))


def fname_clean(key_subtask, mode):
    key = unify_subtask_key(key_subtask)
    return os.path.join(DIR_CLEAN, 'subtask%s_%s.txt'%(key, mode))


def fname_wordcount(key_subtask):
    key = unify_subtask_key(key_subtask)
    return os.path.join(DIR_WORDCOUNT, 'subtask%s.txt'%(key))


def read_data(key_subtask, mode):
    fname = fname_clean(key_subtask, mode)
    lines = []

    with open(fname, 'r') as fobj:
        for line in fobj:
            line = line.strip()
            if line == '':
                continue

            lines.append(line.split('\t'))

    return lines


def read_texts(key_subtask, mode):
    # mode: dev, devtest, train, input
    # key_subtask: A, B, C, D, E
    lines = read_data(key_subtask, mode)

    return map(lambda k: k[-1], lines)


def read_texts_labels(key_subtask, mode):
    lines = read_data(key_subtask, mode)

    return map(lambda k: (k[-2], k[-1]), lines)


def read_wordcount(key_subtask):
    fname = fname_wordcount(key_subtask)
    lines = open(fname, 'r').readlines()
    wc = []
    for line in lines:
        line = line.strip()
        if line == '': continue
        
        params = line.split('\t')
        w = params[0]
        c = int(params[1])

        wc.append((w, c))

    return wc


def read_vocab_topN(key_subtask, n):
    all_wc = read_wordcount(key_subtask)

    # TODO(zxw) return list of top-N frequent words
    wc = map(lambda k:k[0], all_wc)
    return wc    


def read_vocab_minC(key_subtask, min_c):
    wc = read_wordcount(key_subtask)

    # TODO(zxw) return list of words whose count >= min_c
    wc = []
    return wc    

