"""
Once data has been vectorized, this makes large scale adjustments to the data to make it ready for the network.
"""
import numpy as np
from constants import MAX_VECTOR_COUNT
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os
from sklearn.externals import joblib
from common import *

def load_chunk(file_path):
    basename = os.path.basename(file_path)
    is_bot = is_bot_file(file_path)
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



def load_raw(file_path):
    """Loads a raw matrix file with no transformations. Exists mostly as an abstraction."""
    return np.load(file_path)


