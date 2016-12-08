#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: xihao liang
@created: 2016.11.17
"""


def filter(fname_input, fname_output, vocabs):
    vocabs = set(vocabs)

    fobj_output = open(fname_output, 'w')

    with open(fname_input, 'r') as fobj:
        for line in fobj:
            line = line.strip()
            if line == '': continue
        
            loc = line.find(' ')
            if loc == -1: continue

            word = line[:loc]
            if word in vocabs:
                fobj_output.write(line + '\n')

    fobj_output.close()

