"""
This performs all the CSV parsing, wrangling and NLP (word2vec, using SpaCy's large vector set) for the data.
"""
import os
import sys
import emoji
import re
import csv
import spacy
import datetime
import time
import numpy as np
from constants import MAX_VECTOR_COUNT
emojis_list = map(lambda x: ''.join(x.split()), emoji.UNICODE_EMOJI.keys())
r = re.compile('|'.join(re.escape(p) for p in emojis_list))
import logging

root = logging.getLogger()
root.setLevel(logging.DEBUG)

ch = logging.StreamHandler(sys.stdout)
ch.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
root.addHandler(ch)

def filter_emojis(s):
    s = re.sub(r, ' ', s)
    return s


def build_input_arrays(path, nlp):
    output = []

    with open(path, 'r', encoding="utf8") as csvfile:
        logging.info("Tokenizing file at " + path)
        reader = csv.DictReader(csvfile)
        rowCount = 0
        for row in reader:

            rowCount += 1
            if rowCount % 10000 == 0:
                print("Completed a batch of 10000 rows. Current row count: " + str(rowCount))
            rowTokens = []
            no_emojis = filter_emojis(row['text'])
            # we remove emojis primarily because Spacy will not identify them with a word vector, so they're just noise
            tokens = nlp(no_emojis)
            for tok in tokens:
                rowTokens.append(tok.vector)
            output.append(rowTokens)

    tweets = list(filter(lambda x: len(x) != 0, output))
    tweets_filtered = [np.concatenate(x).ravel() for x in tweets]
    tweets_padded = pad_tweet_arr(tweets_filtered)
    logging.info("Completed file {0} and built array of {1} rows.".format(path, len(output)))
    return tweets_padded


def preprocess_input(path):
    """"Removes emojis and writes them out to file."""
    lines = []

    with open(path, 'r') as inputFile:
        for line in inputFile:
            lines.append(line)
    with open(path + "_no_emojis", 'w', encoding="utf8") as outputFile:
        for line in lines:
            outputFile.write(filter_emojis(line))


def write_out_npy_matrix_file(array, filename, chunkCount, doChunks=True):
    np_array = np.array(array)
    if doChunks:
        chunked = np.array_split(np_array, chunkCount)
        chunkCount = 0

        for chunk in chunked:
            logging.info("Writing chunk of size " + str(chunk.size))
            with open(filename + "_" + str(chunkCount) + ".npy", 'wb') as outputFile:
                np_chunk = chunk
                np.save(outputFile, np_chunk)
            chunkCount += 1
    else:
        logging.info("Converting arr into np format...")
        val = np.array(array)
        logging.info("Done. Writing file...")
        with open(filename + ".npy", 'wb') as outputFile:
            np.save(outputFile, val)
        logging.info("Done.")


def unroll_and_filter():
    """My initial formatting choice was terrible. This loads, unrolls, and writes them back out."""
    for fileNo in range(0, 10):
        bot_tweets_unfiltered = []
        bot_loaded = np.load("D:/data/bot_tweets_vectorized_" + str(fileNo) + ".npy")
        bot_tweets_unfiltered = [x[0] for x in bot_loaded]
        bot_tweets = list(filter(lambda x: len(x) != 0, bot_tweets_unfiltered))
        bot_tweets_filtered = [np.concatenate(x).ravel() for x in bot_tweets]
        bot_tweets_padded = pad_tweet_arr(bot_tweets_filtered)
        write_out_npy_matrix_file(bot_tweets_padded, "D:/data/bot_tweets_vectorized_raveled_" + str(fileNo), -1, doChunks=False)
    for fileNo in range(6, 7):
        for chunkNo in range(0, 10):
            regular_tweets_unfiltered = []
            print("Loading " + "D:/data/regular_tweets_vectorized_" + str(fileNo))
            regular_loaded = np.load("D:/data/regular_tweets_vectorized_" + str(fileNo) + "_" + str(chunkNo) + ".npy")
            regular_tweets_unfiltered = np.append(regular_tweets_unfiltered, [x[0] for x in regular_loaded])
            regular_tweets = list(filter(lambda x: len(x) != 0, regular_tweets_unfiltered))
            regular_tweets_filtered = [np.concatenate(x).ravel() for x in regular_tweets]
            regular_tweets_padded = pad_tweet_arr(regular_tweets_filtered)
            write_out_npy_matrix_file(regular_tweets_padded, "D:/data/regular_tweets_vectorized_raveled_" + str(fileNo) + "_" + str(chunkNo), -1, doChunks=False)


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


if __name__ == "__main__":
    # use "python -m spacy download en_core_web_lg" to get the latest vector set
    now = datetime.datetime.now()
    logging.info("Preprocessing begins at " + str(now))
    logging.info("Loading Spacy's large vector set. " + str(now))
    nlp = spacy.load('en_vectors_web_lg')
    now = datetime.datetime.now()
    logging.info("Finished loading. " + str(now))
    # we can load the labels later, since they're trivial to infer from the file we load
    input_path = input("Enter target data directory to traverse: ")
    output_path = input("Enter target output directory for vectorized tweets: ")
    bot_status = input("Are these regular tweets or bot tweets? (0: regular; 1: bot): ")
    matrix_tweets = []
    output_file_number = 0
    prefix_str = "regular_" if bot_status == "0" else "bot_"
    for file in os.listdir(input_path):
        target_path = os.path.join(output_path, prefix_str + str(output_file_number))
        while os.path.exists(target_path):
            target_path = os.path.join(output_path, prefix_str + str(output_file_number))
            logging.info("File {0} exists, skipping.".format(target_path))
            output_file_number += 1
        logging.info("Beginning file " + file)
        matrix_tweets = build_input_arrays(os.path.join(input_path, file), nlp)
        write_out_npy_matrix_file(matrix_tweets, target_path, 10, doChunks=True)
        output_file_number += 1
    now = datetime.datetime.now()
    logging.info("Preprocessing complete at " + str(now))
