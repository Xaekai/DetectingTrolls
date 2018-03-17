import numpy as np
from constants import MAX_VECTOR_COUNT
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from itertools import chain


def process():
    print("Loading data")

    bot_tweets, bot_labels, regular_tweets, regular_labels = load_10_regular_1_bot()
    print("Done loading")
    all_tweets = np.append(regular_tweets, bot_tweets)
    all_labels = np.append(regular_labels, bot_labels)
    del bot_tweets, regular_tweets, regular_labels, bot_labels
    # and now, because doing it in preprocessing would make the files even more ridiculously huge, we're going to pad to 50 with zeroes.
    all_adjusted = pad_tweets(all_tweets)
    del all_tweets
    print("Adjustment complete. Scaling...")
    # split it once on training / test
    print("Scaling data...")
    with open("90_10_tweets" + ".npy", 'wb') as outputFile:
        np.save(outputFile, all_adjusted)
    return
    scaler = StandardScaler(copy=False)
    all_scaled = scaler.fit_transform(all_adjusted)
    print("Scaling complete. Performing model selection...")
    x, x_test, y, y_test = train_test_split(all_scaled, all_labels, test_size=0.25, train_size=0.75, stratify=all_labels)
    # split training again on validation
    x_train, x_cv, y_train, y_cv = train_test_split(x, y, test_size=0.2, train_size=0.8, stratify=y)
    print(len(x_test))
    del x, y
    print("Test size" + str(len(x_test)))
    print("Train size " + str(len(x_train)))
    print("Validation size " + str(len(x_cv)))
    print("Total size " + str(len(all_tweets)))
    print("Test percent " + str(len(x_test) / len(all_tweets)))
    print("Train percent " + str(len(x_train) / len(all_tweets)))
    print("Validation percent " + str(len(x_cv) / len(all_tweets)))
    print("Train test split complete. Converting to np arrays and sending to network.")
    return (np.array(x_test), np.array(x_train), np.array(x_cv), np.array(y_test), np.array(y_train), np.array(y_cv))

def longlist2array(longlist):
    """Allegedly, this is much faster for 2d arrays than asarray."""
    shape = (len(longlist), 12000)
    out = np.zeros(shape)
    for i in range(0, len(longlist)):
        out[i] = longlist[i]
    return out

def get_cv_set():
    print("Loading data")

    bot_tweets, bot_labels, regular_tweets, regular_labels = load_cv_set()
    print("Done loading")
    all_tweets = np.append(regular_tweets, bot_tweets)
    all_labels = np.append(regular_labels, bot_labels)
    all_adjusted = pad_tweets(all_tweets)
    return (np.array(all_adjusted), np.array(all_labels))



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


def load_10_regular_1_bot():
    """Loads files such that roughly 90% are regular and 10% are bot tweets. Takes roughly 6.5gb RAM"""
    bot_tweets_unfiltered = []
    for chunkNo in range(0, 1):
        print("Loading " + "../data/bot_tweets_vectorized" + "_" + str(chunkNo) + ".npy")
        bot_loaded = np.load("../data/bot_tweets_vectorized" + "_" + str(chunkNo) + ".npy")
        bot_tweets_unfiltered = np.append(bot_tweets_unfiltered, [x[0] for x in bot_loaded])
    regular_tweets_unfiltered = []
    for chunkNo in range(0, 3):
        print("Loading " + "../data/regular_tweets_vectorized_1" + "_" + str(chunkNo) + ".npy")
        regular_loaded = np.load("../data/regular_tweets_vectorized_1" + "_" + str(chunkNo) + ".npy")
        regular_tweets_unfiltered = np.append(regular_tweets_unfiltered, [x[0] for x in regular_loaded])
    bot_tweets = list(filter(lambda x: len(x) != 0, bot_tweets_unfiltered))
    regular_tweets = list(filter(lambda x: len(x) != 0, regular_tweets_unfiltered))
    bot_tweets = [np.concatenate(x).ravel() for x in bot_tweets]
    regular_tweets = [np.concatenate(x).ravel() for x in regular_tweets]
    bot_labels = np.ones(len(bot_tweets), dtype=int)
    regular_labels = np.zeros(len(regular_tweets), dtype=int)
    return (bot_tweets, bot_labels, regular_tweets, regular_labels)


def load_cv_set():
    """Loads a subset reserved for cross validation. One bot, one regular."""
    bot_tweets_unfiltered = []
    for chunkNo in range(9, 10):
        print("Loading " + "../data/bot_tweets_vectorized" + "_" + str(chunkNo) + ".npy")
        bot_loaded = np.load("../data/bot_tweets_vectorized" + "_" + str(chunkNo) + ".npy")
        bot_tweets_unfiltered = np.append(bot_tweets_unfiltered, [x[0] for x in bot_loaded])
    regular_tweets_unfiltered = []
    for chunkNo in range(0, 1):
        print("Loading " + "../data/regular_tweets_vectorized_6" + "_" + str(chunkNo) + ".npy")
        regular_loaded = np.load("../data/regular_tweets_vectorized_6" + "_" + str(chunkNo) + ".npy")
        regular_tweets_unfiltered = np.append(regular_tweets_unfiltered, [x[0] for x in regular_loaded])
    bot_tweets = list(filter(lambda x: len(x) != 0, bot_tweets_unfiltered))
    regular_tweets = list(filter(lambda x: len(x) != 0, regular_tweets_unfiltered))
    bot_tweets = [np.concatenate(x).ravel() for x in bot_tweets]
    regular_tweets = [np.concatenate(x).ravel() for x in regular_tweets]
    bot_labels = np.ones(len(bot_tweets), dtype=int)
    regular_labels = np.zeros(len(regular_tweets), dtype=int)
    return (bot_tweets, bot_labels, regular_tweets, regular_labels)


def load_all():
    """WARNING: This takes roughly 45 GB of RAM to do in entirety."""
    """Totally impractical right now without something like Spark"""
    bot_tweets_unfiltered = []
    regular_tweets_unfiltered = []
    for fileNo in range(0, 9):
        print("Loading " + "../data/bot_tweets_vectorized_" + str(fileNo) + ".npy")
        bot_loaded = np.load("../data/bot_tweets_vectorized_" + str(fileNo) + ".npy")
        bot_tweets_unfiltered = np.append(bot_tweets_unfiltered, [x[0] for x in bot_loaded])
        for chunkNo in range(0, 9):
            print("Loading " + "../data/regular_tweets_vectorized_" + str(fileNo) + "_" + str(chunkNo) + ".npy")
            regular_loaded = np.load("../data/regular_tweets_vectorized_" + str(fileNo) + "_" + str(chunkNo) + ".npy")
            regular_tweets_unfiltered = np.append(regular_tweets_unfiltered, [x[0] for x in regular_loaded])
    bot_tweets = list(filter(lambda x: len(x) != 0, bot_tweets_unfiltered))
    regular_tweets = list(filter(lambda x: len(x) != 0, regular_tweets_unfiltered))

    bot_labels = np.ones(len(bot_tweets), dtype=int)
    regular_labels = np.zeros(len(regular_tweets), dtype=int)
    return (bot_tweets, bot_labels, regular_tweets, regular_labels)


if __name__ == "__main__":
    process()
