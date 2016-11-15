#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: xihao liang
@created: 2016.11.14
"""

class Indexer:
    def __init__(self, labels):
        self._labels = labels
        self._inv = dict(map(
                        lambda a: (a[1], a[0]),
                        enumerate(labels)
                    ))
        self._n_label = len(labels)

    def idx(self, label):
        if isinstance(label, list):
            return [self._inv[lbl] for lbl in label if lbl in self._inv]
        
        if isinstance(label, str):
            label = label.decode("utf8")
        
        return self._inv.get(label, None)

    def label(self, idx):
        return self._labels[idx] if idx < self._n_label else None


def test():
    indexer = Indexer(list('ABC'))
    print indexer.idx('B')
    print indexer.label(2)

if __name__ == '__main__':
    test()
