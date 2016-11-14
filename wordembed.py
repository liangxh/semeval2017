#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: xihao liang
@created: 2016.11.14
"""

import os
import numpy as np
import data_manager

SEED = 111
np.random.seed(SEED)


def get(vocabs, fname_Wemb, dim):
    """get the weight matrix for word-embedding

    Args:
        vocabs: list of vocabularies(str)
        fname_Wemb: filename of the original embedding vectors under DIR_WEMB
        dim: dimension of the embedding vector

    Return:
        list of NumPy.ndarray, each of the array refers to an embedding vector,
        which corresponds to $vocabs
    """

    vocab_dict = dict(map(lambda k: (k[1], k[0]), enumerate(vocabs)))
    vecs = [None for i in range(len(vocabs))]

    n_vocab = len(vocabs)
    count = 0

    path_Wemb = os.path.join(data_manager.DIR_WEMB, fname_Wemb)
    with open(path_Wemb, 'r') as fobj:
        for line in fobj:
            line = line.strip()

            loc = line.find(' ')
            if loc == -1: continue
            
            vocab = line[:loc]
            if vocab in vocab_dict:
                idx = vocab_dict[vocab]
                vecs[idx] = np.array(map(float, line.split(' ')[1:]))

                count += 1
                if count == n_vocab:
                    break

    n_notsupported = n_vocab - count
    rand_vecs = (np.random.random((n_notsupported, dim)) - 0.5) * 0.01
    idx = 0

    for i, vec in enumerate(vecs):
        if vec is None:
            vecs[i] = rand_vecs[idx]
            idx += 1
            if idx == n_notsupported:
                break

    return vecs


def test():
    vocabs = data_manager.read_vocabs_topN('A', 4000)
    vecs = get(vocabs, 'glove.twitter.27B.25d.txt', 25)

    print vecs[:10]


if __name__ == '__main__':
    test()
