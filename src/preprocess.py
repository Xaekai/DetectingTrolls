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

def writeOutNpyMatrixFile(array, filename, max_size_per_chunk):
    np_array = np.array(array)
    chunked = np.array_split(np_array, 10)
    chunkCount = 0
    for chunk in chunked:
        print("Writing chunk of size " + str(chunk.size))
        with open(filename + "_" + str(chunkCount) + ".npy", 'wb') as outputFile:
            np_chunk = chunk
            np.save(outputFile, np_chunk)
        chunkCount += 1

if __name__ == "__main__":
    now = datetime.datetime.now()
    print("Preprocessing begins at " + str(now))
    print("Loading Spacy's large vector set. " + str(now))
    nlp = spacy.load('en_vectors_web_lg')
    now = datetime.datetime.now()
    print("Finished loading. " + str(now))

    bot_tweets = buildInputArrays('../data/bot_tweets.csv', nlp)
    # bot_labels = np.ones(len(bot_tweets), dtype=int)
    # we can load the labels later, since they're trivial to infer from the file we load
    writeOutNpyMatrixFile(bot_tweets, "bot_tweets_vectorized", 25000)
    # a bit brutish but I don't want to deal with a lot of file IO
    for chunkNumber in range(0, 8):
        print("Beginning file " + str(chunkNumber))
        matrix_tweets = []
        matrix_tweets += buildInputArrays('../data/non_bot_chunk_' + str(chunkNumber) + '.csv', nlp)
        writeOutNpyMatrixFile(matrix_tweets, "D:/data/" + "regular_tweets_vectorized_" + str(chunkNumber), 25000)
    now = datetime.datetime.now()
    print("Preprocessing complete at " + str(now))

