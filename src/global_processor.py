"""
Once data has been vectorized, this makes large scale adjustments to the data to make it ready for the network.
"""
import numpy as np
from constants import MAX_VECTOR_COUNT
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os
from sklearn.externals import joblib
import logging
import sys
import random
import itertools as IT

def load_chunk(file_path):
    basename = os.path.basename(file_path)
    if "bot" in basename:
        logging.info("Flagging {0} as a bot file.".format(file_path))
        is_bot = True
    elif "regular" in basename:
        is_bot = False
        logging.info("Flagging {0} as a regular file.".format(file_path))
    else:
        logging.error("File {0} is neither regular or bot!".format(file_path))
        return
    tweets = np.load(file_path)
    labels = []
    scaler = joblib.load('scaler.pkl')
    all_scaled = scaler.transform(tweets)
    if is_bot:
        labels = np.ones(len(all_scaled), dtype=int)
    else:
        labels = np.zeros(len(all_scaled), dtype=int)
    x_train, x_test, y_train, y_test = train_test_split(all_scaled, labels, shuffle=True, test_size=0.2)
    logging.info("X Train size:" + str(len(x_train)))
    logging.info("X Test size:" + str(len(x_test)))
    logging.info("Y Train size:" + str(len(y_train)))
    logging.info("Y Train size:" + str(len(y_test)))

    return np.array(x_train), np.array(x_test), np.array(y_train), np.array(y_test)




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