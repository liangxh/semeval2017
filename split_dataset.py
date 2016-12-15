#! /usr/bin/env python
# -*- coding: utf-8 -*-


def split():
    f_train = open('../data/clean/emo_tweet_en_train.txt', 'w')
    f_dev = open('../data/clean/emo_tweet_en_dev.txt', 'w')
    # f_test = open('../data/clean/emo_tweet_en_test.txt', 'w')

    f = open('../data/clean/emo_tweet_en_all.txt', 'r')

    lines = f.readlines()
    num = len(lines)

    for i in range(num):
        if i < num * 0.7:
            f_train.write(lines[i])
        else:
            f_dev.write(lines[i])

    f.close()

if __name__ == '__main__':
    split()
