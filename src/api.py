# FORCE CPU USAGE - this VPS does not have a GPU
# Note: When I loaded this using AVX2 instructions I got a strange error.
# probably because it's a virtual CPU
# in any case, if you have the same problem, install this TF whl 
# https://github.com/fo40225/tensorflow-windows-wheel/tree/master/1.6.0/py36/CPU/sse2
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = ""


from flask import Flask, abort, request
import flask
import spacy
import datetime
import time
import numpy as np
import emoji
import re
import logging

from constants import MAX_VECTOR_COUNT
from sklearn.externals import joblib
import keras
from keras.models import load_model
# set up logging
logging.basicConfig(filename='latest.log', level=logging.DEBUG)
# Init- must occur in global scope, since other funcs will need them later
emojis_list = map(lambda x: ''.join(x.split()), emoji.UNICODE_EMOJI.keys())
r = re.compile('|'.join(re.escape(p) for p in emojis_list))
app = Flask(__name__)
now = datetime.datetime.now()
print("Preprocessing begins at " + str(now))
print("Loading SpaCy's large vector set. " + str(now))
nlp = spacy.load('en_vectors_web_lg')
now = datetime.datetime.now()
print("Finished loading. " + str(now))
print("Loading scaler from trained data.")
scaler = joblib.load("scaler.pkl")
print("Loading complete.")
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
    
    
@app.route('/predict', methods=['POST'])
def vector():
    data = request.get_json()
    
    if not data:
        logging.error("An invalid request was submitted to the API.")
        abort(400)
    target = data['target']
    logging.info("Processing new request " + target)
    processed = process_tweet(target)
    scaled = scale_tweet(processed)
    predicted = predict_scaled_tweet(scaled)
    return flask.jsonify(predicted.tolist())

def main():
    #app.run(debug=False, use_reloader=False)
    app.run(debug=False, use_reloader=False, host="0.0.0.0")
main()  
    
