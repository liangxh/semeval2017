#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: xiwen zhao
@created: 2016.12.5
"""


def get_test_labels():
    for subtask in [ 'BD', 'CE']:
        f_labels = open('../data/DOWNLOAD/Subtask_%s/twitter-2016test-%s.txt' % (subtask, subtask), 'r')
        f_unknowns = open('../data/test sets/SemEval2016-task4-test.subtask-%s.txt' % subtask, 'r')

        f_new = open('../data/test sets/subtask%s_test_new.txt' % subtask, 'w')

        f_labels = f_labels.readlines()
        f_unknowns = f_unknowns.readlines()

        for line_label, line_unknown in zip(f_labels, f_unknowns):
            line_label = line_label.strip().split('\t')
            line_unknown = line_unknown.strip().split('\t')

            if subtask == 'A':  # id not matched for A
                continue

            elif subtask == 'BD':
                if (line_label[0], line_label[1]) == (line_unknown[0], line_unknown[1]):
                    f_new.write(line_label[0] + '\t' + line_label[1] + '\t' + line_label[-1] + '\t' + line_unknown[-1] + '\n')

            else:
                if line_label[1] == line_unknown[1]:
                    f_new.write(line_label[0] + '\t' + line_label[1] + '\t' + line_label[-1] + '\t' + line_unknown[-1] + '\n')

if __name__ == '__main__':
    get_test_labels()
