# -*- coding: utf8 -*-
"""
@author: xiwen zhao
@created: 2016.11.19
"""

from util import tweet


def split_subtask(input_fname, output_fname):  # labels for BCDE --> BD, CE
    f = open(output_fname, 'w')
    texts = open(input_fname, 'r').read()[:-1].split('\n')
    for text in texts:
        items = text.split('\t')
        if items[-1] == 'Not Available':
            continue
        '''
        if items[-2] == '1' or items[-2] == '2':
            items[-2] = 'negative'

            items[-1] = tweet.preprocess(items[-1])
            f.write('\t'.join(items)+'\n')
        elif items[-2] == '4' or items[-2] == '5':
            items[-2] = 'positive'

            items[-1] = tweet.preprocess(items[-1])
            f.write('\t'.join(items)+'\n')
        '''
        if items[-2] == '1':
            items[-2] = '-2'
        elif items[-2] == '2':
            items[-2] = '-1'
        elif items[-2] == '3':
            items[-2] = '0'
        elif items[-2] == '4':
            items[-2] = '1'
        elif items[-2] == '5':
            items[-2] = '2'

        items[-1] = tweet.preprocess(items[-1])
        f.write('\t'.join(items)+'\n')

    f.close()


if __name__ == '__main__':
    split_subtask('../data/raw/trialCDE.txt', '../data/clean/trialCE.txt')
