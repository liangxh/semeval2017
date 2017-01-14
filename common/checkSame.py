#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: xiwen zhao
@created: 2017.01.14
"""

import os


def checkSame():
    path = '../../data/DOWNLOAD/Subtask_A'  # data after preprocessed
    output_name = '../../data/clean/subtaskBD_previous.txt'
    f_output = open(output_name, 'w')
    num_BD = []

    for mode in (['train_dev', 'devtest', 'test_new']):
        BD_lines = open('../../data/clean/subtaskBD_%s.txt' % mode, 'r').readlines()

        for line in BD_lines:
            line = line.strip().split('\t')
            if line[0] not in num_BD:
                num_BD.append(line[0])

    print len(num_BD)

    for root, dirs, files in os.walk(path):
        for fname in files:
            if fname != '.DS_Store':
                A_name = os.path.join(root, fname)
                A_lines = open(A_name, 'r').readlines()

                for line in A_lines:
                    line = line.strip().split('\t')
                    if line[1] == 'neutral':
                        continue

                    if line[0] not in num_BD:
                        num_BD.append(line[0])
                        f_output.write('\t'.join(line) + '\n')

    print len(num_BD)


if __name__ == '__main__':
    checkSame()