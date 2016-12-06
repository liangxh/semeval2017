#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: xiwen zhao
@created: 2016.12.6
"""


def comb_train_dev():
    for subtask in ['A', 'BD', 'CE']:
        f_train = open('../../data/clean/subtask%s_train.txt' % subtask, 'r').readlines()
        f_dev = open('../../data/clean/subtask%s_dev.txt' % subtask, 'r').readlines()

        f_train_dev = open('../../data/clean/subtask%s_train_dev.txt' % subtask, 'w')

        for line_train in f_train:
            f_train_dev.write(line_train)
        for line_dev in f_dev:
            f_train_dev.write(line_dev)
        '''
        if subtask is not 'A':
            for line_train in f_train:
                line_train = line_train.strip().split('\t')
                for line_dev in f_dev:
                    line_dev = line_dev.strip().split('\t')
                    if line_train[1] == line_dev[1]:
                        print 'Overlapping topics'
        '''


if __name__ == '__main__':
    comb_train_dev()
