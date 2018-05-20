import re
import emoji
import logging
import sys
import numpy as np
from constants import *
import os
import random
import itertools as IT

emojis_list = map(lambda x: ''.join(x.split()), emoji.UNICODE_EMOJI.keys())
r = re.compile('|'.join(re.escape(p) for p in emojis_list))
root = logging.getLogger()
root.setLevel(logging.DEBUG)
ch = logging.StreamHandler(sys.stdout)
ch.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
root.addHandler(ch)


def pad_array(A, size):
    t = size - len(A)
    return np.pad(A, pad_width=(0, t), mode='constant')


def pad_tweet_arr(arr):
    out = []
    for tweet in arr:
        if len(tweet) > MAX_VECTOR_COUNT:
            logging.warning("Trimming tweet vectors with vector length " + str(len(tweet)))
            tweet = tweet[:MAX_VECTOR_COUNT]
            # trim, but it's highly unlikely there are that many tokens
        else:
            tweet = pad_array(tweet, MAX_VECTOR_COUNT)
        out.append(tweet)
    return out


def filter_emojis(s):
    s = re.sub(r, ' ', s)
    return s


def is_bot_file(file_path):
    is_bot = None
    basename = os.path.basename(file_path)
    if "bot" in basename:
        logging.info("Flagging {0} as a bot file.".format(file_path))
        is_bot = True
    elif "regular" in basename:
        is_bot = False
        logging.info("Flagging {0} as a regular file.".format(file_path))
    else:
        logging.error("File {0} is neither regular or bot!".format(file_path))
    return is_bot


def get_interleaved_file_list(data_dir):
    """Builds an interleaved file list such that bot entries are evenly spaced"""
    files = os.listdir(data_dir)
    bot_files = [f for f in files if "bot" in f]
    regular_files = [f for f in files if "regular" in f]
    random.shuffle(bot_files)
    random.shuffle(regular_files)
    interleaved_file_list = evenly_spaced(bot_files, regular_files)
    logging.info("Interleaved file list is {0}".format(interleaved_file_list))
    return interleaved_file_list

def evenly_spaced(*iterables):
    """
    >>> evenly_spaced(range(10), list('abc'))
    [0, 1, 'a', 2, 3, 4, 'b', 5, 6, 7, 'c', 8, 9]
    """
    return [item[1] for item in
            sorted(IT.chain.from_iterable(
            zip(IT.count(start=1.0 / (len(seq) + 1),
                         step=1.0 / (len(seq) + 1)), seq)
            for seq in iterables))]
