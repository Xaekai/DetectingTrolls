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
logging.basicConfig(filename='latest.log', level=logging.DEBUG)
# Init- must occur in global scope, since other funcs will need them later
emojis_list = map(lambda x: ''.join(x.split()), emoji.UNICODE_EMOJI.keys())
r = re.compile('|'.join(re.escape(p) for p in emojis_list))
application = app = Flask(__name__)
now = datetime.datetime.now()
print("Preprocessing begins at " + str(now))
print("Loading SpaCy's large vector set. " + str(now))
nlp = spacy.load('en_vectors_web_lg')
now = datetime.datetime.now()
print("Finished loading. " + str(now))
print("Loading scaler from trained data.")
scaler = joblib.load("scaler.pkl")
print("Loading complete.")

# twitter api setup
print("Establishing Twitter API connection.")
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth)

print("Loading Keras model...")
model = load_model("troll_model.h5")

print("Load complete.")
print("Listening for requests...")
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

    
def tokenizeTweet(tweet):
    output = []
    rowTokens = []
    tokens = nlp(tweet)
    for tok in tokens:
        output.append(tok.vector)
    return output

    
def process_tweet(tweet):
    tokens = tokenizeTweet(filterEmojis(tweet))
    raveled = np.concatenate(tokens).ravel()
    padded = pad_tweet(raveled)
    return padded
    
    
def scale_tweet(tweet):
    return scaler.transform(tweet.reshape(1, -1))
    
    
def filterEmojis(s):
    s = re.sub(r, ' ', s)
    return s

    
def predict_scaled_tweet(tweet):
    return model.predict(tweet)
    
    
@app.route('/', methods=['GET'])
def heartbeat():
    return "<html><body>Alive and well!</body></html>"
    
    
@app.route('/predict', methods=['GET'])
def vector():
    target = request.args.get('target')
    if not target:
        logging.error("An invalid request was submitted to the API.")
        abort(400)
    
    logging.info("Processing new request " + target)
    final_target = "" #the text the AI will actually see; retrieved from APIs if url
    if validators.url(target):
        if "twitter" in get_tld(target):
            logging.info("Flagged " + target + " as a tweet.")
            final_target = get_tweet_raw_text(api, target)
        else:
            logging.error("A url was submitted " + target + " but it is not a TLD matching the API filter.")
            abort(406) #406 not accepted
    else:
        final_target = target
    processed = process_tweet(final_target)
    scaled = scale_tweet(processed)
    predicted = predict_scaled_tweet(scaled)
    return flask.jsonify(predicted.tolist())

    
def get_tweet_raw_text(api, url):
    idVal = url.split('/')[-1]
    public_tweet = api.get_status(idVal, tweet_mode='extended')
    logging.info("Retrieved tweet: " + public_tweet.full_text + " from url " + url)
    return public_tweet.full_text
    
    
def main():
    #app.run(debug=False, use_reloader=False)
    app.run(debug=False, use_reloader=False, host="127.0.0.1")
if __name__ == '__main__':
    main()

    
