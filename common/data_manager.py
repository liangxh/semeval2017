#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: xiwen zhao
@created: 2016.11.13
"""

import os

DIR_DATA = os.path.join(os.path.dirname(__file__), '../../data')
DIR_RAW = os.path.join(DIR_DATA, 'raw')
DIR_CLEAN = os.path.join(DIR_DATA, 'clean')
DIR_WORDCOUNT = os.path.join(DIR_DATA, 'wordcount')
DIR_COMMON = os.path.join(DIR_DATA, 'common')
DIR_WEMB = os.path.join(DIR_DATA, 'wemb')

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

    return map(lambda k: (k[-1], k[-2]), lines)


def read_id_label(key_subtask):  # only for subtask A
    fname = fname_clean(key_subtask, 'devtest')
    lines = open(fname, 'r').readlines()
    id_label = []
    for line in lines:
        if line == '': continue

        params = line.strip().split('\t')
        id_label.append((params[0], params[1]))

    return id_label


def read_id_topic_label(key_subtask):  # for subtask B, C, D, E
    fname = fname_clean(key_subtask, 'devtest')
    lines = open(fname, 'r').readlines()
    id_topic_label = []
    for line in lines:
        if line == '': continue

        params = line.strip().split('\t')
        id_topic_label.append((params[0], params[1], params[2]))

    return id_topic_label


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

    return wc  # list of tuple


def read_vocabs(key_subtask):
    all_wc = read_wordcount(key_subtask)

    vocabs = map(lambda k: k[0], all_wc)  # list of str
    return vocabs    


def read_vocabs_topN(key_subtask, n):
    vocabs = read_vocabs(key_subtask)

    if n > len(vocabs):
        vocabs = vocabs[:n]

    return vocabs


def read_vocabs_minC(key_subtask, min_c):
    all_wc = read_wordcount(key_subtask)

    for i in range(len(all_wc)):
        if all_wc[i][1] >= min_c:
            continue
        break
    wc = all_wc[:i]
    wc = map(lambda k: k[0], wc)
    return wc



