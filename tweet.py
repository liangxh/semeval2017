#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: # TODO(zxw) make it your code
@created: 2016.11.10
@TODO(zxw) add more functions for different processes
"""

import re

pattern_space = re.compile(r'\s+')  # s->blank space
pattern_http = re.compile(r'https?://\S+')
pattern_punct = re.compile(r'[!\"\$&\'\(\)\*\+,\-\./:;=\?@\[\\\]\^_`\{\|\}~]')
# keep # for hashtags, and keep < > for specific tokens like <LINK>
pattern_mention = re.compile(r'@\S+')
pattern_shorten = re.compile(r'')

def merge_space(text):
    """turn several spaces into one space"""
    return pattern_space.sub(' ', text)

def replace_url(text, token = '<LINK>'):
    """replace http:// or https:// in $text with $token"""
    return pattern_http.sub(token, text)

def replace_mention(text, token = '<MENTION>'):
    """replace @USERNAME in $text with $token"""
    return pattern_mention.sub(token, text)

def remove_punct(text):
    """remove punctuation in $text"""
    return pattern_punct.sub(' ', text)

def shorten_word(text):
    """shorten elongated words to a maximum of three character repetitions"""
    return pattern_shorten.sub()

def preprocess(text):
    """preprocess tweet so that unnecessary tokens are deleted
    and certain tokens are replaced by unique tokens, etc.

    Args:
        text: a string, content of a tweet
    
    Returns:
        a string, the processed tweet
    """
    text = text.lower()
    text = replace_url(text)
    text = replace_mention(text)
    #text = remove_punct(text)

    # must be done at last
    text = merge_space(text).strip()

    return text

def test():
    """
    use this function to process certain sample if you keep failing
    """
    text = '@lxh http://www.baidu.com \'\"hello,,,, how re             you?????'
    print preprocess(text)

def main():
    """
    use this function to test your function interactively
    """
    while True:
        text = raw_input("say something: ").strip()
        text = text.encode("utf8")
        if text == 'q':
            break;
        
        print preprocess(text)

if __name__ == '__main__':
    """
    you can only switch between this two modes manually
    """
    main()
    #test()

