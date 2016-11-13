#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: 
@created:
"""
#TODO(zxw) fill in the fields

import tweet
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

    #TODO(zxw) change twitter_tokenize
    for text in texts:
        tokens = tokenizer.twitter_tokenize(text.strip())
        for token in tokens:
            token = token.decode("utf8")
            if token in wc:
                wc[token] += 1
            else:
                wc[token] = 1

    return wc


def count_singletext(text):
    wc = {}
    #TODO(zxw) implement this function

    tokens = tokenizer.twitter_tokenize(text.strip())
    print tokens  # [u'hello', u',', u'123', u'this', u'is', u'me', u'.', u':)', u':(', u'(', u'(', u'!', u'!', u'!']
    for token in tokens:
        token = token.decode("utf8")
    for token in tokens:
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

    # TODO(zxw) implement this function
    f = open(fname_output, 'w')
    wc = sorted(wc, key=lambda t: (-t[1], t[0]))
    for w, c in wc:
        line = '%s\t%d\n'%(w, c)
        f.write(line)
    f.close()

def main():
    #TODO(zxw) implement this function

    wc = {}
    for fname_input, fname_clean in zip(fnames_input, fnames_clean):
        clean(fname_input, fname_clean)
        texts = read_texts(fname_clean)
        wc = count(texts, wc)


    export(wc.items(), 'wordcount.txt')

if __name__ == '__main__':
    main()

