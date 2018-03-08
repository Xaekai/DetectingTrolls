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

def writeOutFile(array, filename, max_terms_per_file):
    


if __name__ == "__main__":
    now = datetime.datetime.now()
    print("Preprocessing begins at " + str(now))
    print("Loading Spacy's large vector set. " + str(now))
    nlp = spacy.load('en_vectors_web_lg')
    now = datetime.datetime.now()
    print("Finished loading. " + str(now))

    bot_tweets = buildInputArrays('../data/bot_tweets.csv', nlp)
    #bot_labels = np.ones(len(bot_tweets), dtype=int)
    #we can load the labels later, since they're trivial to infer from the file we load
    with open("bot_vectorized.npy", 'wb') as bot_tweet_file:
        bot_tweet_np = np.array(bot_tweets)
        np.save(bot_tweet_file, bot_tweet_np)

    # a bit brutish but I don't want to deal with a lot of file IO
    for chunkNumber in range(0, 7):
        print("Beginning file " + str(chunkNumber))
        matrix_tweets = []
        matrix_tweets += buildInputArrays('../data/non_bot_chunk_' + str(chunkNumber) + '.csv', nlp)
        with open("ordinary_vectorized_" + str(chunkNumber) + ".npy", 'wb') as ordinary_tweet_file:
            ordinary_tweet_np = np.array(matrix_tweets)
            np.save(ordinary_tweet_file, ordinary_tweet_np)
    now = datetime.datetime.now()
    print("Preprocessing complete at " + str(now))

