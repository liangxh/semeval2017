# -*- coding: utf8 -*-
"""
@author:
@created: 2016.11.13
"""

# TODO(zxw) fill in the fields

import tweet

def clean(input_fname, output_fname):
    f = open(output_fname, 'w')
    texts = open(input_fname, 'r').read()[:-1].split('\n')
    for text in texts:
        items = text.split('\t')
        items[-1] = tweet.preprocess(items[-1])
        f.write('\t'.join(items[1:])+'\n')
    f.close()


