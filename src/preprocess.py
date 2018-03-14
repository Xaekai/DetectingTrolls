import emoji
import re
import csv
import spacy
import datetime
import time
import numpy as np

import cnn
emojis_list = map(lambda x: ''.join(x.split()), emoji.UNICODE_EMOJI.keys())
r = re.compile('|'.join(re.escape(p) for p in emojis_list))
MAX_VECTOR_COUNT = 40*300

def filterEmojis(s):
    s = re.sub(r, ' ', s)
    return s


def buildInputArrays(path, nlp):
    output = []

    with open(path, 'r', encoding="utf8") as csvfile:
        print("Tokenizing file at " + path)
        reader = csv.DictReader(csvfile)
        rowCount = 0
        for row in reader:
            rowCount += 1
            if rowCount % 10000 == 0:
                print("Completed a batch of 10000 rows. Current row count: " + str(rowCount))
            rowTokens = []
            tokens = nlp(row['text'])
            for tok in tokens:
                rowTokens.append(tok.vector)
            output.append((rowTokens, row['userid']))
    print(datetime.datetime.now())
    print(len(output))
    return output


def preprocessInput(path):
    """"Removes emojis and writes them out to file."""
    lines = []

    with open(path, 'r') as inputFile:
        for line in inputFile:
            lines.append(line)
    with open(path + "_no_emojis", 'w', encoding="utf8") as outputFile:
        for line in lines:
            outputFile.write(filterEmojis(line))

def writeOutNpyMatrixFile(array, filename, chunkCount, doChunks=True):
    np_array = np.array(array)
    if doChunks:
        chunked = np.array_split(np_array, chunkCount)
        chunkCount = 0

        for chunk in chunked:
            print("Writing chunk of size " + str(chunk.size))
            with open(filename + "_" + str(chunkCount) + ".npy", 'wb') as outputFile:
                np_chunk = chunk
                np.save(outputFile, np_chunk)
            chunkCount += 1
    else:
        val = np.array(array)
        with open(filename + ".npy", 'wb') as outputFile:
            np.save(outputFile, val)
def unrollAndFilter():
    """My initial formatting choice was terrible. This loads, unrolls, and writes them back out."""
    for fileNo in range(0, 10):
        bot_tweets_unfiltered = []
        bot_loaded = np.load("D:/data/bot_tweets_vectorized_" + str(fileNo) + ".npy")
        bot_tweets_unfiltered = [x[0] for x in bot_loaded]
        bot_tweets = list(filter(lambda x: len(x) != 0, bot_tweets_unfiltered))
        bot_tweets_filtered = [np.concatenate(x).ravel() for x in bot_tweets]
        bot_tweets_padded = pad_tweet_arr(bot_tweets_filtered)
        writeOutNpyMatrixFile(bot_tweets_padded, "D:/data/bot_tweets_vectorized_raveled_" + str(fileNo), -1, doChunks=False)
    for fileNo in range(6, 7):
        for chunkNo in range(0, 10):
            regular_tweets_unfiltered = []
            print("Loading " + "D:/data/regular_tweets_vectorized_" + str(fileNo))
            regular_loaded = np.load("D:/data/regular_tweets_vectorized_" + str(fileNo) + "_" + str(chunkNo) + ".npy")
            regular_tweets_unfiltered = np.append(regular_tweets_unfiltered, [x[0] for x in regular_loaded])
            regular_tweets = list(filter(lambda x: len(x) != 0, regular_tweets_unfiltered))
            regular_tweets_filtered = [np.concatenate(x).ravel() for x in regular_tweets]
            regular_tweets_padded = pad_tweet_arr(regular_tweets_filtered)
            writeOutNpyMatrixFile(regular_tweets_padded, "D:/data/regular_tweets_vectorized_raveled_" + str(fileNo) + "_" + str(chunkNo), -1, doChunks=False)

def padarray(A, size):
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
            tweet = padarray(tweet, MAX_VECTOR_COUNT)
        out.append(tweet)
    return out

if __name__ == "__main__":
    # use "python -m spacy download en_core_web_lg" to get the latest vector set
    now = datetime.datetime.now()
    print("Preprocessing begins at " + str(now))
    print("Loading Spacy's large vector set. " + str(now))
    nlp = spacy.load('en_vectors_web_lg')
    now = datetime.datetime.now()
    print("Finished loading. " + str(now))

    # bot_tweets = buildInputArrays('../data/bot_tweets.csv', nlp)
    # bot_labels = np.ones(len(bot_tweets), dtype=int)
    # we can load the labels later, since they're trivial to infer from the file we load
    # writeOutNpyMatrixFile(bot_tweets, "bot_tweets_vectorized", 10)
    # a bit brutish but I don't want to deal with a lot of file IO
    for chunkNumber in range(0, 7):
        print("Beginning file " + str(chunkNumber))
        matrix_tweets = []
        matrix_tweets += buildInputArrays('../data/non_bot_chunk_' + str(chunkNumber) + '.csv', nlp)
        writeOutNpyMatrixFile(matrix_tweets, "regular_tweets_vectorized_" + str(chunkNumber), 10)
    now = datetime.datetime.now()
    print("Preprocessing complete at " + str(now))
