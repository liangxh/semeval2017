#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: xihao liang
@created: 2016.11.13
"""

import sys
import re
REPEAT_MIN = 3
pattern_rep = re.compile(r'((?P<repw>[a-z])(?P=repw){%d,})'%(REPEAT_MIN - 1))

import data_manager

def main():
    key_subtask = sys.argv[1]
    wc = data_manager.read_wordcount(key_subtask)
    vocabs = set(map(lambda k:k[0], wc))

    for w, c in wc:
        m = pattern_rep.search(w)
        if m:
            f1 = shorten(w, 2) in vocabs
            f2 = shorten(w, 1) in vocabs
            print '%20s'%w, '%3d'%c, f1, f2, not (f1 and f2)

def shorten(text, min_count = REPEAT_MIN):
    res = pattern_rep.findall(text)
    for m in res:
        text = text.replace(m[0], m[0][:min_count]) 
    
    return text


def test():
    a = 'aaaaaab'
    print shorten(a)

if __name__ == '__main__':
    main()
    #test()
