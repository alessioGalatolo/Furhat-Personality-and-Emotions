#!/usr/bin/python
# -*- coding: utf-8 -*-
#
#   /$$$$$$  /$$      /$$ /$$      /$$ /$$$$$$$$
#  /$$__  $$| $$$    /$$$| $$$    /$$$|__  $$__/
# | $$  \__/| $$$$  /$$$$| $$$$  /$$$$   | $$   
# |  $$$$$$ | $$ $$/$$ $$| $$ $$/$$ $$   | $$   
#  \____  $$| $$  $$$| $$| $$  $$$| $$   | $$   
#  /$$  \ $$| $$\  $ | $$| $$\  $ | $$   | $$   
# |  $$$$$$/| $$ \/  | $$| $$ \/  | $$   | $$   
#  \______/ |__/     |__/|__/     |__/   |__/  
#
#
# Developed during Biomedical Hackathon 6 - http://blah6.linkedannotation.org/
# Authors: Ramya Tekumalla, Javad Asl, Juan M. Banda
# Contributors: Kevin B. Cohen, Joanthan Lucero

import re
import emoji


def remove_emoticons(text):
    emoticon_pattern = re.compile(r"""
    (?:
      [<>]?
      [:;=8]                     # eyes
      [\-o\*\']?                 # optional nose
      [\)\]\(\[dDpP/\:\}\{@\|\\] # mouth
      |
      [\)\]\(\[dDpP/\:\}\{@\|\\] # mouth
      [\-o\*\']?                 # optional nose
      [:;=8]                     # eyes
      [<>]?
    )""")
    return emoticon_pattern.sub(r'', text)


def remove_at(text):
    at_pattern = re.compile(r"""(?:@[\w_]+)""", flags=re.UNICODE)
    return at_pattern.sub(r'', text)


def remove_hashtags(text):
    hashtag_pattern = re.compile(r"""(?:\#+[\w_]+[\w\'_\-]*[\w_]+)""", flags=re.UNICODE)
    return hashtag_pattern.sub(r'', text)


def remove_emoji(text):
    emoji_pattern = re.compile("["
                        u"\U0001F600-\U0001F64F"  # emoticons
                        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                        u"\U0001F680-\U0001F6FF"  # transport & map symbols
                        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                        u"\U00002702-\U000027B0"
                        u"\U000024C2-\U0001F251"
                        "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)


def remove_urls(text):
    result = re.sub(r"http\S+", "", text)
    return(result)


def remove_twitter_urls(text):
    clean = re.sub(r"pic.twitter\S+", "",text)
    return(clean)


def give_emoji_free_text(text):
    return emoji.get_emoji_regexp().sub(r'', text)


def preprocess_text(text):
    text = text.replace('\n', '')
    text = text.replace('\r', '')

    text = remove_urls(text)
    text = remove_twitter_urls(text)
    text = remove_emoticons(text)
    text = remove_at(text)
    text = remove_hashtags(text)
    text = give_emoji_free_text(text)

    return text
