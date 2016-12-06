#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: xihao liang
@created: 2016.12.3
"""

import os
import commands
import numpy as np
from optparse import OptionParser

from common import data_manager


def load_topic_probs(fname):
    topic_probs = {}
    with open(fname, 'r') as fobj:
        for line in fobj:
            line = line.replace('\n', '')
            if line == '': continue
            
            params = line.split('\t')
            topic = params[0]
            probs = map(float, params[1:])

            if not topic in topic_probs: topic_probs[topic] = []
            topic_probs[topic].append(probs)

    return topic_probs.items()


def argmax_dist(probs, ydim):
    dist = np.zeros(ydim)

    for prob in probs:
        dist[np.argmax(prob)] += 1

    return dist


def sum_dist(probs, ydim):
    dist = np.zeros(ydim)
    
    for prob in probs:
        dist += np.asarray(prob)

    return dist


def sqr_dist(probs, ydim):
    dist = np.zeros(ydim)
    for prob in probs: dist += np.asarray(prob) ** 2
    return dist


def calculate_dist(probs, func_dist = argmax_dist):
    if len(probs[0]) == 1:  # subtask D
        probs = map(lambda p: [p[0], 1. - p[0]], probs)

    ydim = len(probs[0])
    dist = func_dist(probs, ydim)

    dist /= np.sum(dist)
    return dist


def main():
    optparser = OptionParser()
    optparser.add_option('-t', '--subtask', dest='key_subtask', default='D')
    optparser.add_option('-s', '--dataset', dest='key_mode', default='test_new')
    optparser.add_option('-m', '--model', dest='model_name', default='finki')
    optparser.add_option('-d', '--dist_type', dest='dist_type', default=None)
    opts, args = optparser.parse_args()

    if not opts.dist_type:
        func_dist = argmax_dist
    else:
        try:
            func_dist = eval('%s_dist' % opts.dist_type)
        except:
            print 'undefined dist_type'
            return

    fname = os.path.join(data_manager.DIR_PRED_PROB, '%s_%s_%s_new.txt'%(opts.key_subtask, opts.key_mode, opts.model_name))

    topic_probs = load_topic_probs(fname)  # list of tuple (str, list of probs)

    fname_output = os.path.join(data_manager.DIR_RESULT, '%s_%s_%s_dist_new.txt'%(opts.key_subtask, opts.key_mode, opts.model_name))

    fobj_output = open(fname_output, 'w')
    for topic, probs in topic_probs:
        dist = calculate_dist(probs, func_dist)

        fobj_output.write('%s\t%s\n'%(
            topic,
            '\t'.join(map(lambda f: '%.12f' % f, dist))
        ))

    fobj_output.close()
    o = commands.getoutput(
        "perl eval/score-semeval2016-task4-subtask%s.pl " \
         "../data/result/%s_%s_gold.txt %s" % (
            opts.key_subtask,
            opts.key_subtask, opts.key_mode,
            fname_output,
        )
    )

    try:
        o = o.strip()
        lines = o.split("\n")
        score = float(lines[-1].split("\t")[-1])
        print score
    except:
        print "calcdist: [warning] invalid output file for semeval measures tool"
        print [o, ]


if __name__ == '__main__':
    main()
