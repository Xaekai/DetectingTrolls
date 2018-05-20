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
import logging
import sys
from common import *


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


def preprocess_data(input_data, output_path):
    for input_entry in input_data:
        matrix_tweets = []
        output_file_number = 0
        prefix_str = "regular_" if input_entry[1] == "0" else "bot_"
        logging.info("Prefix is {0}".format(prefix_str))
        logging.info("Processing a new input file!")
        input_path = input_entry[0]
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

if __name__ == "__main__":
    # use "python -m spacy download en_core_web_lg" to get the latest vector set
    now = datetime.datetime.now()
    logging.info("Preprocessing begins at " + str(now))
    logging.info("Loading Spacy's large vector set. " + str(now))
    nlp = spacy.load('en_vectors_web_lg')
    now = datetime.datetime.now()
    logging.info("Finished loading. " + str(now))
    # we can load the labels later, since they're trivial to infer from the file we load
    input_data = []
    should_continue = True
    while should_continue:
        input_path = input("Enter target data directory to traverse: ")
        bot_status = input("Are these regular tweets or bot tweets? (0: regular; 1: bot): ")
        input_data.append((input_path, bot_status))
        choice = input("Enter another? y/n ")
        if not choice == "y":
            should_continue = False
    logging.info("Preparing to traverse {0} directories.".format(len(input_data)))
    output_path = input("Enter target output directory for vectorized tweets: ")
    preprocess_data(input_data, output_path)

