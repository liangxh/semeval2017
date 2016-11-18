#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: xihao liang
@created: 2016.11.13
"""

import os
import sys
import re
REPEAT_MIN = 3
pattern_rep = re.compile(r'((?P<repw>[a-z])(?P=repw){%d,})'%(REPEAT_MIN - 1))
pattern_vocab = re.compile(r'^([a-zA-Z\']+)$')

rep_char = set(list('eoprsg'))

import data_manager

vocabref = set(
                open(os.path.join(
                    data_manager.DIR_COMMON, 'vocabref.txt'
                ), 'r').read()[:-1].split('\n')
            )

def shorten(text, min_count = REPEAT_MIN):
    if not pattern_vocab.match(text) or text in vocabref:
        return None

    res = pattern_rep.findall(text)
    if len(res) == 0: return None
 
    for i in range(1, min_count):
        count = min_count - i
        vocab = text
        for m in res:
            if m[0][0] in rep_char:
                vocab = vocab.replace(m[0], m[0][:count])
            else:          
                vocab = vocab.replace(m[0], m[0][0])
        
        if vocab in vocabref:
            if vocab.endswith('iing') and not text.endswith('skiing') and not text.endswith('taxiing'):
                vocab = vocab.replace('iing', 'ing')

            return vocab

    return vocab


def main():
    key_subtask = sys.argv[1]
    wc = data_manager.read_wordcount(key_subtask)
    vocabs = set(map(lambda k:k[0], wc))

    dict_wc = dict(wc)

    for w, c in wc:
        w_s = shorten(w)
        if w_s:
            print '%s (%d) -> %s (%s)'%(w, c, w_s, dict_wc.get(w_s, None))

def test():
    a = 'aaaaaab'
    print shorten(a)

if __name__ == '__main__':
    main()
    #test()
