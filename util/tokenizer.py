#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: xihao liang
@created: 2016.11.13
@see: Interface for different tokenizers
"""

def simple_tokenize(text):
    return text.split(" ")

def twitter_tokenize(text):
    """tokenize a sentence with  NLTK Twitter Tokenizer"""
    if isinstance(text, str): text = text.decode("utf8")

    global tknzr
    return tknzr.tokenize(text)

# tokenize = simple_tokenize

from nltk.tokenize import TweetTokenizer
tokenize = twitter_tokenize; tknzr = TweetTokenizer()

def test():
    while True:
        text = raw_input("say something: ")
        if text == "q":
            break
        #print type(tokenize(text.strip()))
        print ' / '.join(tokenize(text.strip()))

    print "see you~"


if __name__ == '__main__':
    test()
