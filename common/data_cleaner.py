# -*- coding: utf8 -*-
"""
@author: xiwen zhao
@created: 2016.11.13
"""

from util import tweet

def clean(input_fname, output_fname):
    f = open(output_fname, 'w')
    texts = open(input_fname, 'r').read()[:-1].split('\n')
    for text in texts:
        items = text.split('\t')
        if items[-1] == 'Not Available':
            continue

        items[-1] = tweet.preprocess(items[-1])
        f.write('\t'.join(items)+'\n')
    f.close()


