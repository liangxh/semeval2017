#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: xiwen zhao
@created: 2016.11.13
"""

import tokenizer


def count(texts, wc = None):
    """count the frequency of distinct tokens

    Args:
        texts: list of string

    Return:
        A dictionary, whose keys refer to the tokens and
        values refer to the frequency of the respective
        token. 
    """

    if wc is None:
        wc = {}

    for text in texts:
        tokens = tokenizer.tokenize(text.strip())
        for token in tokens:
            token = token.decode("utf8")
            if token in wc:
                wc[token] += 1
            else:
                wc[token] = 1

    return wc


def export(wc, fname_output):
    """Export wordcount into an output file.
    Each line contain a token and a integer which is
    the frequency of the token.

    Args:
        wc: list of tuple(str, int), wordcount
        fname_output: filename for the output file
    """

    f = open(fname_output, 'w')
    wc = sorted(wc, key=lambda t: (-t[1], t[0]))
    for w, c in wc:
        line = '%s\t%d\n'%(w, c)
        f.write(line)
    f.close()

