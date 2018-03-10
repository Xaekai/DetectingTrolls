import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split


def process():
    print("Loading data")

    regular_tweets = []
    bot_tweets = []
    for fileNo in range(0, 9):
        print("Loading " + "D:/data/bot_tweets_vectorized_" + str(fileNo) + ".npy")
        bot_loaded = np.load("D:/data/bot_tweets_vectorized_" + str(fileNo) + ".npy")
        bot_tweets = np.append(bot_tweets, [(x[0], 1) for x in bot_loaded])
        for chunkNo in range(0, 9):
            print("Loading " + "D:/data/regular_tweets_vectorized_" + str(fileNo) + "_" + str(chunkNo) + ".npy")
            regular_loaded = np.load("D:/data/regular_tweets_vectorized_" + str(fileNo) + "_" + str(chunkNo) + ".npy")
            regular_tweets = np.append(regular_tweets, [(x[0], 1) for x in regular_loaded])
    print("Done loading")
    all_tweets = np.append(regular_tweets, bot_tweets)
    # split it once on training / test
    x, x_test, y, y_test = train_test_split(all_tweets, test_size=0.25, train_size=0.75)
    # split training again on validation
    x_train, x_cv, y_train, y_cv = train_test_split(x, y, test_size=0.2, train_size=0.8)

    print("Test size" + str(len(x_test)))
    print("Train size " + str(len(x_train)))
    print("Validation size " + str(len(x_cv)))
    print("Total size " + str(len(data)))
    print("Test percent " + str(len(x_test) / len(data)))
    print("Train percent " + str(len(x_train) / len(data)))
    print("Validation percent " + str(len(x_cv) / len(data)))
    return (x, x_test, x_train, x_cv, y, y_test, y_train, y_cv)


if __name__ == "__main__":
    process()