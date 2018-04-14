# FORCE CPU USAGE - this VPS does not have a GPU
# Note: When I loaded this using AVX2 instructions I got a strange error.
# probably because it's a virtual CPU
# in any case, if you have the same problem, install this TF whl 
# https://github.com/fo40225/tensorflow-windows-wheel/tree/master/1.6.0/py36/CPU/sse2
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = ""

# generic
from flask import Flask, abort, request
import flask

import datetime
import time
import numpy as np
import emoji
import re
import logging

# twitter related
import tweepy
import validators
from tld import get_tld
from oauth import consumer_key, consumer_secret, access_token, access_token_secret
# keras / ai related
from constants import MAX_VECTOR_COUNT
from sklearn.externals import joblib
import spacy
import keras
from keras.models import load_model


# set up logging
logFormatter = logging.Formatter("%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s")
log = logging.getLogger()

fileHandler = logging.FileHandler("latest.log")
fileHandler.setFormatter(logFormatter)
log.addHandler(fileHandler)

consoleHandler = logging.StreamHandler()
consoleHandler.setFormatter(logFormatter)
log.addHandler(consoleHandler)

# Init- must occur in global scope, since other funcs will need them later
emojis_list = map(lambda x: ''.join(x.split()), emoji.UNICODE_EMOJI.keys())
r = re.compile('|'.join(re.escape(p) for p in emojis_list))
def padarray(A, size):
    t = size - len(A)
    return np.pad(A, pad_width=(0, t), mode='constant')


def pad_tweet(tweet):
    if len(tweet) > MAX_VECTOR_COUNT:
        print("Trimming tweet vectors with vector length " + str(len(tweet)))
        tweet = tweet[:MAX_VECTOR_COUNT]
        # trim, but it's highly unlikely there are that many tokens
    else:
        # we need to add fake empty tokens to pad
        needed_empty = MAX_VECTOR_COUNT - len(tweet)
        tweet = padarray(tweet, MAX_VECTOR_COUNT)
    return tweet

    
def tokenizeTweet(nlp, tweet):
    output = []
    rowTokens = []
    tokens = nlp(tweet)
    for tok in tokens:
        output.append(tok.vector)
    return output

    
def tokenize_ravel_pad(tweet):
    tokens = tokenizeTweet(filterEmojis(tweet))
    raveled = np.concatenate(tokens).ravel()
    padded = pad_tweet(raveled)
    return padded
    
    
def scale_tweet(scaler, tweet):
    return scaler.transform(tweet.reshape(1, -1))
    
    
def filterEmojis(s):
    s = re.sub(r, ' ', s)
    return s

    
def predict_scaled_tweet(model, tweet):
    return model.predict(tweet)
    
    
def get_tweet_raw_text(api, url):
    idVal = url.split('/')[-1]
    public_tweet = api.get_status(idVal, tweet_mode='extended')
    logging.info("Retrieved tweet: " + public_tweet.full_text + " from url " + url)
    return public_tweet.full_text

def get_tweet_raw_text(api, url):
    idVal = url.split('/')[-1]
    public_tweet = api.get_status(idVal, tweet_mode='extended')
    logging.info("Retrieved tweet: " + public_tweet.full_text + " from url " + url)
    return public_tweet.full_text    

def main():
    app.run(debug=False, use_reloader=False, host="127.0.0.1")
if __name__ == '__main__':
    main()

    
