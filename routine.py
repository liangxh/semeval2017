#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: xiwen zhao
@created: 2016.11.13
"""

import os
import sys
import re
import types
from common import data_manager


def clean():
    """$ ./routine.py clean"""

    from common import data_cleaner
    
    for root, dirs, files in os.walk(data_manager.DIR_RAW):
        for fname in files:
            input_fname = os.path.join(root, fname)
            output_fname = os.path.join(root.replace('raw', 'clean'), re.sub('\.\w+$', '.txt', fname))
            print "cleaning file %s" % fname
            data_cleaner.clean(input_fname, output_fname)


def countword():
    """$ ./routine.py countword KEY_SUBTASK"""
    '''
    if len(sys.argv) < 3:
        print "required arguments: KEY_SUBTASK"
        return
    '''
    from util import wordcount

    # key_subtask = sys.argv[2].upper()

    wc = {}
    for key_subtask in list('BCDE'):
        for mode in ["train", "dev", "devtest", "test_new"]:
            texts = data_manager.read_texts(key_subtask, mode)
            wc = wordcount.count(texts, wc)

        output_fname = data_manager.fname_wordcount(key_subtask)
        wordcount.export(wc.items(), output_fname)


def countword_emo():
    """$ ./routine.py countword_emo"""
    from util import wordcount

    wc = {}
    texts = data_manager.read_emo_texts('all_cut')
    wc = wordcount.count(texts, wc)

    output_fname = '../data/wordcount/emo_tweet.txt'
    wordcount.export(wc.items(), output_fname)


def gold():
    """$ ./routine.py gold"""
    
    from common import gold_builder
    for key_subtask in list('BCDE'):
        for mode in ['train', 'dev', 'devtest', 'test_new']:
            gold_builder.build(key_subtask, mode)


funcs = {}
for name in dir():
    if not name.startswith('_'): # not builtin-functions
        func = eval(name)
        if isinstance(func, types.FunctionType):
            funcs[func.__name__] = func


def get_help():
    global funcs

    for func in funcs.values():
        print "[function] %s\n[usage] %s\n"%(func.__name__, func.__doc__ if func.__doc__ else '<missing>')


def main():
    if len(sys.argv) < 2:
        get_help()
    else:
        func_name = sys.argv[1]    
        funcs.get(func_name, get_help)()


if __name__ == '__main__':
    main()
