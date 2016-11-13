#! /usr/bin/env 
# -*- coding: utf-8 -*-
"""
@author: 
@created: 
"""
# TODO(zxw) fill in the fields

import os

DIR_DATA = os.path.join(os.path.dirname(__file__), '../data')
DIR_RAW = os.path.join(DIR_DATA, 'raw')
DIR_CLEAN = os.path.join(DIR_DATA, 'clean')
DIR_WORDCOUNT = os.path.join(DIR_DATA, 'wordcount')

def unify_subtask_key(key):
    key = key.upper()

    if key in ["B", "D"]: return "BD"
    if key in ["C", "E"]: return "CD"
    
    return key


def fname_raw(key_subtask, mode):
    key = unify_subtask_key(key_subtask)
    return os.path.join(DIR_RAW, 'subtask%s-%s.tsv'%(key, mode))


def fname_clean(key_subtask, mode):
    key = unify_subtask_key(key_subtask)
    return os.path.join(DIR_CLEAN, 'subtask%s-%s.txt'%(key, mode))


def fname_wordcount(key_subtask):
    key = unify_subtask_key(key_subtask)
    return os.path.join(DIR_WORDCOUNT, 'subktask%s.txt'%(key))


def read_data(key_subtask, mode):
    fname = fname_clean(key_subtask, mode)
    lines = []

    with open(fname, 'r') as fobj:
        for line in fobj:
            line = line.strip()
            if line == '': continue

            lines.append(line.split('\t'))

    return lines


def read_texts(key_subtask, mode):
    lines = read_data(key_subtask, mode)

    return map(lambda k:k[-1], lines)


def read_texts_labels(key_subtask, mode):
    # TODO(zxw) implement this function like read_texts
    return []

