#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: xiwen zhao
@created: 2016.11.13
"""

import os
import input_adapter

DIR_DATA = os.path.join(os.path.dirname(__file__), '../../data')
DIR_RAW = os.path.join(DIR_DATA, 'raw')
DIR_CLEAN = os.path.join(DIR_DATA, 'clean')
DIR_WORDCOUNT = os.path.join(DIR_DATA, 'wordcount')
DIR_COMMON = os.path.join(DIR_DATA, 'common')
DIR_WEMB = os.path.join(DIR_DATA, 'wemb')
DIR_MODEL = os.path.join(DIR_DATA, 'model')
DIR_RESULT = os.path.join(DIR_DATA, 'result')
DIR_PRED_PROB = os.path.join(DIR_DATA, 'pred_prob')


def unify_subtask_key(key):
    key = key.upper()

    if key in ["B", "D"]: return "BD"
    if key in ["C", "E"]: return "CE"
    
    return key


def fname_raw(key_subtask, mode):
    key = unify_subtask_key(key_subtask)
    return os.path.join(DIR_RAW, 'subtask%s_%s.tsv'%(key, mode))


def fname_clean(key_subtask, mode):
    key = unify_subtask_key(key_subtask)
    return os.path.join(DIR_CLEAN, 'subtask%s_%s.txt'%(key, mode))


def fname_clean_emo(key_subtask, mode):
    return os.path.join(DIR_CLEAN, 'emo_tweet_en_%s.txt' % mode)


def fname_wordcount(key_subtask):
    key = unify_subtask_key(key_subtask)
    return os.path.join(DIR_WORDCOUNT, 'subtask%s.txt'%(key))


def fname_model_weight(key_subtask, model_name):
    return os.path.join(DIR_MODEL, 'subtask%s_%s_weight_new.hdf5'%(key_subtask, model_name))


def fname_model_config(key_subtask, model_name):
    return os.path.join(DIR_MODEL, 'subtask%s_%s_config_new.json'%(key_subtask, model_name))


def fname_gold(key_subtask, mode):
    return os.path.join(DIR_RESULT, '%s_%s_gold.txt'%(key_subtask, mode))


def fname_pred(key_subtask, mode):
    return os.path.join(DIR_RESULT, '%s_%s_pred.txt'%(key_subtask, mode))


def read_data(key_subtask, mode):
    fname = fname_clean(key_subtask, mode)
    lines = []

    with open(fname, 'r') as fobj:
        for line in fobj:
            line = line.strip()
            if line == '':
                continue

            lines.append(line.split('\t'))

    return lines


def read_emo_texts_labels(key_subtask, mode):
    fname = fname_clean_emo(key_subtask, mode)
    lines = []

    with open(fname, 'r') as fobj:
        for line in fobj:
            line = line.strip()
            if line == '':
                continue

            lines.append(line.split('\t'))

    return map(lambda k: (k[-1], k[-2]), lines)


def read_texts(key_subtask, mode):
    # mode: dev, devtest, train, input
    # key_subtask: A, B, C, D, E
    lines = read_data(key_subtask, mode)

    return map(lambda k: k[-1], lines)


def read_texts_labels(key_subtask, mode):
    lines = read_data(key_subtask, mode)

    return map(lambda k: (k[-1], k[-2]), lines)


def read_topic(key_subtask, mode):
    if key_subtask is not 'A':
        lines = read_data(key_subtask, mode)

        return map(lambda k: k[1], lines)
    return None


def write_id_label(key_subtask, pred_classes):  # only for subtask A
    fname = fname_clean(key_subtask, 'devtest')
    lines = open(fname, 'r').readlines()
    id_label = []
    for line in lines:
        if line == '': continue

        params = line.strip().split('\t')
        id_label.append((params[0], params[1]))

    tweet_id = []
    tweet_label = []

    for item in id_label:
        tweet_id.append(item[0])
        tweet_label.append(item[1])

    f_pred = open('../data/result/pred_result%s.txt' % key_subtask, 'w')

    for t_id, result in zip(tweet_id, pred_classes):
        f_pred.write(t_id + '\t' + input_adapter.get_label_indexer(key_subtask).label(result) + '\n')
    f_pred.close()

    f_gold = open('../data/result/gold_result%s.txt' % key_subtask, 'w')

    for t_id, t_label in zip(tweet_id, tweet_label):
        f_gold.write(t_id + '\t' + t_label + '\n')
    f_gold.close()


def write_id_topic_label(key_subtask, pred_classes):  # for subtask B, C
    fname = fname_clean(key_subtask, 'devtest')
    lines = open(fname, 'r').readlines()
    id_topic_label = []
    for line in lines:
        if line == '': continue

        params = line.strip().split('\t')
        id_topic_label.append((params[0], params[1], params[2]))

    tweet_id = []
    tweet_topic = []
    tweet_label = []

    for item in id_topic_label:
            tweet_id.append(item[0])
            tweet_topic.append(item[1])
            tweet_label.append(item[2])

    f_pred = open('../data/result/pred_result%s.txt' % key_subtask, 'w')

    for t_id, t_topic, result in zip(tweet_id, tweet_topic, pred_classes):
        f_pred.write(t_id + '\t' + t_topic + '\t' + input_adapter.get_label_indexer(key_subtask).label(result) + '\n')
    f_pred.close()

    f_gold = open('../data/result/gold_result%s.txt' % key_subtask, 'w')

    for t_id, t_topic, t_label in zip(tweet_id, tweet_topic, tweet_label):
        f_gold.write(t_id + '\t' + t_topic + '\t' + t_label + '\n')
    f_gold.close()


def write_topic_label(key_subtask, pred_classes):  # for subtask D
    fname = fname_clean(key_subtask, 'devtest')
    lines = open(fname, 'r').readlines()
    topic_label = []
    for line in lines:
        if line == '': continue

        params = line.strip().split('\t')
        topic_label.append((params[1], params[2]))

    topics = map(lambda k:k[0],topic_label)

    dict_topic = {}  # dict: topic --> [labels in devtest]
    for item in topic_label:
        if item[0] in dict_topic:
            dict_topic[item[0]].append(item[1])
        else:
            dict_topic[item[0]] = [item[1], ]

    # gold file
    prob = {}  # dict of topic and relative probabilities in devtest (gold file)
    for topic in dict_topic.keys():
        pos = 0
        tweet_num = len(dict_topic.get(topic, -1))
        for label in dict_topic.get(topic, -1):
            if label == 'positive':
                pos += 1

        p = round(float(pos) / tweet_num, 12)
        prob[topic] = (p, 1. - p)

    f_gold = open('../data/result/gold_result%s.txt' % key_subtask, 'w')

    for t_topic, t_gold_prob in prob.items():
        f_gold.write(t_topic + '\t' + str(t_gold_prob[0]) + '\t' + str(t_gold_prob[1]) + '\t' +
                     str(len(dict_topic.get(t_topic, -1))) + '\n')
    f_gold.close()

    # prediction file
    dict_topic = {}
    for topic, pred_class in zip(topics, pred_classes):
        if topic in dict_topic:
            dict_topic[topic].append(pred_class)
        else:
            dict_topic[topic] = [pred_class, ]

    prob = {}  # dict of topic and relative probabilities in prediction file
    for topic, labels in dict_topic.items():
        pos = len([None for label in labels if label == 1])

        p = float(pos) / len(labels)
        prob[topic] = (p, 1. - p)

    f_pred = open('../data/result/pred_result%s.txt' % key_subtask, 'w')

    for t_topic, t_pred_prob in prob.items():
        f_pred.write(t_topic + '\t' + str(t_pred_prob[0]) + '\t' + str(t_pred_prob[1]) + '\n')
    f_pred.close()


def write_topic_5labels(key_subtask, pred_classes):  # for subtask E
    fname = fname_clean(key_subtask, 'devtest')
    lines = open(fname, 'r').readlines()
    topic_label = []
    for line in lines:
        if line == '': continue

        params = line.strip().split('\t')
        topic_label.append((params[1], params[2]))

    topics = map(lambda k:k[0], topic_label)

    dict_topic = {}  # dict: topic --> [labels in devtest]
    for item in topic_label:
        if item[0] in dict_topic:
            dict_topic[item[0]].append(item[1])
        else:
            dict_topic[item[0]] = [item[1], ]

    # gold file
    prob = {}  # dict of topic and relative probabilities in devtest (gold file)
    for topic in dict_topic.keys():
        hneg = neg = neu = pos = hpos = 0
        tweet_num = len(dict_topic.get(topic, -1))
        for label in dict_topic.get(topic, -1):
            if label == '-2':
                hneg += 1
            elif label == '-1':
                neg += 1
            elif label == '0':
                neu += 1
            elif label == '1':
                pos += 1
            else: hpos += 1

        hn = float(hneg) / tweet_num
        n = float(neg) / tweet_num
        u = float(neu) / tweet_num
        p = float(pos) / tweet_num
        hp = float(hpos) / tweet_num
        prob[topic] = (hn, n, u, p, hp)

    f_gold = open('../data/result/gold_result%s.txt' % key_subtask, 'w')

    for t_topic, t_gold_prob in prob.items():
        line ='%s\t%.12f\t%.12f\t%.12f\t%.12f\t%.12f\t%d\n'%(t_topic, t_gold_prob[0], t_gold_prob[1], t_gold_prob[2],
                                                    t_gold_prob[3], t_gold_prob[4], len(dict_topic.get(t_topic, -1)))
        f_gold.write(line)
    f_gold.close()

    # prediction file
    dict_topic = {}
    for topic, pred_class in zip(topics, pred_classes):
        if topic in dict_topic:
            dict_topic[topic].append(pred_class)
        else:
            dict_topic[topic] = [pred_class, ]

    prob = {}  # dict of topic and relative probabilities in prediction file
    for topic, labels in dict_topic.items():
        hneg = len([None for label in labels if label == 0])
        neg = len([None for label in labels if label == 1])
        neu = len([None for label in labels if label == 2])
        pos = len([None for label in labels if label == 3])
        hpos = len([None for label in labels if label == 4])

        hn = float(hneg) / len(labels)
        n = float(neg) / len(labels)
        u = float(neu) / len(labels)
        p = float(pos) / len(labels)
        hp = float(hpos) / len(labels)
        prob[topic] = (hn, n, u, p, hp)

    f_pred = open('../data/result/pred_result%s.txt' % key_subtask, 'w')

    for t_topic, t_pred_prob in prob.items():
        line ='%s\t%.12f\t%.12f\t%.12f\t%.12f\t%.12f\n'%(t_topic, t_pred_prob[0], t_pred_prob[1], t_pred_prob[2],
                                                    t_pred_prob[3], t_pred_prob[4])
        f_pred.write(line)
    f_pred.close()


def read_wordcount(key_subtask):
    fname = fname_wordcount(key_subtask)
    lines = open(fname, 'r').readlines()
    wc = []
    for line in lines:
        line = line.strip()
        if line == '': continue
        
        params = line.split('\t')
        w = params[0]
        c = int(params[1])

        wc.append((w, c))

    return wc  # list of tuple


def read_vocabs(key_subtask):
    all_wc = read_wordcount(key_subtask)

    vocabs = map(lambda k: k[0], all_wc)  # list of str
    return vocabs    


def read_vocabs_topN(key_subtask, n):
    vocabs = read_vocabs(key_subtask)

    if n > len(vocabs):
        vocabs = vocabs[:n]

    return vocabs


def read_vocabs_minC(key_subtask, min_c):
    all_wc = read_wordcount(key_subtask)

    for i in range(len(all_wc)):
        if all_wc[i][1] >= min_c:
            continue
        break
    wc = all_wc[:i]
    wc = map(lambda k: k[0], wc)
    return wc



