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
from oauth import consumer_key, consumer_secret, access_token, access_token_secret
import tweepy
# common module code
from common import *
log.info("Common module loaded!")
# twitter streamer
from streamer import *

import validators
from tld import get_tld

# keras / ai related
from constants import MAX_VECTOR_COUNT
from sklearn.externals import joblib
import spacy
from keras.models import load_model


application = app = Flask(__name__)
now = datetime.datetime.now()
print("Preprocessing begins at " + str(now))
print("Loading SpaCy's large vector set. " + str(now))
nlp = spacy.load('en_vectors_web_lg')
now = datetime.datetime.now()
print("Finished loading. " + str(now))
log.info("Loading scaler from trained data.")
scaler = joblib.load("scaler.pkl")
log.info("Loading complete.")
# twitter api setup

log.info("Loading Keras model...")
model = load_model("troll_model.h5")
log.info("Load complete.")
log.info("Initializing twitter feed stream on new thread...")
log.info("Establishing Twitter API connection.")
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth)
log.info("Establishing model complete.")
botStreamListener = BotStreamer(model, scaler)
stream = tweepy.Stream(auth, botStreamListener)
# stream.filter(track=['@Troll_Spotter'], async=True)
log.info("Async thread creation complete!")

@app.route('/', methods=['GET'])
def heartbeat():
    return "<html><body>Alive and well!</body></html>"
    
    
@app.route('/predict', methods=['GET'])
def vector():
    target = request.args.get('target')
    if not target:
        log.error("An invalid request was submitted to the API.")
        abort(400)
    
    log.info("Processing new request " + target)
    final_target = "" #the text the AI will actually see; retrieved from APIs if url
    if validators.url(target):
        if "twitter" in get_tld(target):
            log.info("Flagged " + target + " as a tweet.")
            if api.rate_limit_status() >= 0:
                final_target = get_tweet_raw_text(api, target)
            else:
                abort(420)
        else:
            log.error("A url was submitted " + target + " but it is not a TLD matching the API filter.")
            abort(406) #406 not accepted
    else:
        final_target = target
    tokenized_raveled_padded = tokenize_ravel_pad(final_target)
    scaled = scale_tweet(scaler, tokenize_ravel_padded)
    predicted = predict_scaled_tweet(model, scaled)
    return flask.jsonify(predicted.tolist())

    

    
    
def main():
    #app.run(debug=False, use_reloader=False)
    app.run(debug=False, use_reloader=False, host="127.0.0.1")
if __name__ == '__main__':
    main()

    
