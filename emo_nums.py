#! /usr/bin/env python
# -*- coding: utf-8 -*-


import sys
reload(sys)
sys.setdefaultencoding('utf8')

def emo_labels():
    f_emo = open('../data/clean/emo_tweet_en_all.txt', 'r')
    lines = f_emo.readlines()
    emo_num = {}

    for line in lines:
        line = line.decode('utf8')
        line = line.strip().split('\t')
        if line[0] in emo_num.keys():
            emo_num[line[0]] += 1
        else:
            emo_num[line[0]] = 1

    f_out = open('../data/clean/emo_nums.txt', 'w')
    f_cut = open('../data/clean/emo_cut.txt', 'w')    

    sum_num = 0
    emo_num = sorted(emo_num.items(), key = lambda k:-k[1])
    for (emo, num) in emo_num:
        if num < 10000: 
            f_cut.write('%s\t%s\n' % (emo, num))
        else:
            sum_num += num
            f_out.write('%s\t%s\n' % (emo, num))
    
    f_emo.close()
    print sum_num


if __name__ == '__main__':
    emo_labels()
