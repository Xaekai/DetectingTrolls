"""
Once data has been vectorized, this makes large scale adjustments to the data to make it ready for the network.
"""

import numpy as np
from constants import MAX_VECTOR_COUNT
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import normalize
from itertools import chain


def build_large_arr(base_name, bot_no, chunk_lower, chunk_higher, fileNo):
    """Completes preprocessing by loading in a few chunks, normalizing them across zero mean
    and saving them to npy files. The network will create and load these.
    """
    print("Loading data")
    print("")
    bot_tweets, bot_labels, regular_tweets, regular_labels = load_chunk(bot_no, chunk_lower, chunk_higher, fileNo)
    print("Done loading")
    all_tweets = np.append(regular_tweets, bot_tweets)
    all_labels = np.append(regular_labels, bot_labels)
    del bot_tweets, regular_tweets, regular_labels, bot_labels
    all_adjusted = pad_tweets(all_tweets)
    del all_tweets
    print("Adjustment complete. Scaling...")
    # split it once on training / test
    print("Saving data...")
    with open("../data/" + base_name +"_tweets" + ".npy", 'wb') as outputFile:
        np.save(outputFile, all_adjusted)
    with open("../data/" + base_name + "_labels" + ".npy", 'wb') as outputFile:
        np.save(outputFile, all_labels)


def process(large_file_path_to_load, labels_path):
    print("Loading data from large file...")
    all_adjusted = np.load(large_file_path_to_load + ".npy")
    all_labels = np.load(labels_path + ".npy")

    print("Loaded. Scaling data...")
    scaler = StandardScaler(copy=False)
    all_scaled = scaler.fit_transform(all_adjusted)
    print("Scaling complete. Performing model selection...")
    x_train, x_test, y_train, y_test = train_test_split(all_scaled, all_labels, shuffle=False, test_size=0.2)
    print("X Train size:" + str(len(x_train)))
    print("X Test size:" + str(len(x_test)))
    print("Y Train size:" + str(len(y_train)))
    print("Y Train size:" + str(len(y_test)))
    print("Train test split complete. Converting to np arrays and sending to network.")
    return x_train, x_test, y_train, y_test
def process_no_split(large_file_path_to_load, labels_path):
    print("Loading data from large file...")
    all_adjusted = np.load(large_file_path_to_load + ".npy")
    all_labels = np.load(labels_path + ".npy")

    print("Loaded. Scaling data...")
    scaler = StandardScaler(copy=False)
    all_scaled = scaler.fit_transform(all_adjusted)
    print("Scaling complete. Sending to validator...")
    return all_scaled, all_labels

def pad_tweets(all_tweets):
    all_adjusted = []
    for tweet in all_tweets:
        if len(tweet) >  MAX_VECTOR_COUNT:
            print("Trimming tweet vectors with vector length " + str(len(tweet)))
            tweet = tweet[:MAX_VECTOR_COUNT]
            # trim, but it's highly unlikely there are that many tokens
        else:
            # we need to add fake empty tokens to pad
            tweet = pad_array(tweet, MAX_VECTOR_COUNT)
        all_adjusted.append(tweet)
    return all_adjusted


def pad_array(A, size):
    t = size - len(A)
    return np.pad(A, pad_width=(0, t), mode='constant')


def pad_tweet_arr(arr):
    out = []
    for tweet in arr:
        if len(tweet) > MAX_VECTOR_COUNT:
            print("Trimming tweet vectors with vector length " + str(len(tweet)))
            tweet = tweet[:MAX_VECTOR_COUNT]
            # trim, but it's highly unlikely there are that many tokens
        else:
            # we need to add fake empty tokens to pad
            needed_empty = MAX_VECTOR_COUNT - len(tweet)
            tweet = pad_array(tweet, MAX_VECTOR_COUNT)
        out.append(tweet)
    return out


def load_chunk(bot_no, chunk_lower, chunk_higher, fileNo):
    """Loads files such that roughly 90% are regular and 10% are bot tweets. Takes roughly 6.5gb RAM"""
    bot_tweets_unfiltered = []
    print("Loading " + "../data/bot_tweets_vectorized" + "_" + str(bot_no) + ".npy")
    bot_loaded = np.load("../data/bot_tweets_vectorized" + "_" + str(bot_no) + ".npy")
    bot_tweets_unfiltered = np.append(bot_tweets_unfiltered, [x[0] for x in bot_loaded])
    regular_tweets_unfiltered = []
    for chunkNo in range(chunk_lower, chunk_higher):
        print("Loading " + "../data/regular_tweets_vectorized_" + str(fileNo) + "_" + str(chunkNo) + ".npy")
        regular_loaded = np.load("../data/regular_tweets_vectorized_" + str(fileNo) + "_" + str(chunkNo) + ".npy")
        regular_tweets_unfiltered = np.append(regular_tweets_unfiltered, [x[0] for x in regular_loaded])
    bot_tweets = list(filter(lambda x: len(x) != 0, bot_tweets_unfiltered))
    regular_tweets = list(filter(lambda x: len(x) != 0, regular_tweets_unfiltered))
    bot_tweets = [np.concatenate(x).ravel() for x in bot_tweets]
    regular_tweets = [np.concatenate(x).ravel() for x in regular_tweets]
    bot_labels = np.ones(len(bot_tweets), dtype=int)
    regular_labels = np.zeros(len(regular_tweets), dtype=int)
    return (bot_tweets, bot_labels, regular_tweets, regular_labels)
