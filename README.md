# Classifying Russian Troll Tweets via Artificial Neural Nets

## Data
NBC Universal has provided 200,000 tweets determined to be related in some way to Russian trolls. You can find the dataset [here](nbcnews.com/tech/social-media/now-available-more-200-000-deleted-russian-troll-tweets-n844731?cid=sm_npd_nn_tw_ma)

I combined this with a dataset of 1.6m "regular" tweets located (here)[http://thinknook.com/twitter-sentiment-analysis-training-corpus-dataset-2012-09-22/]. 

## Goal
The hope is that we will be able to classify tweets as Russian troll tweets given this training set. I don't expect to see anything even close to 80% accuracy, but if it can help find the needle in the haystack, it's a net positive.
It's also a nice educational exercise in data wrangling and neural nets.

## Implementation (to be updated)

### Preprocessing

This step parses existing data using the `csv` library in Python, `numpy` and a few other tools. Most notably, it **strips out the emojis**. This is important because otherwise when we attempt to perform NLP on it, 
there will be no useful vectors for the emojis. Plus, emojis seem to consistently break a variety of Python tools and I'd rather not deal with it. 

I used AlexDel's [csv_splitter](https://gist.github.com/AlexDel) and made a few modifications to make it more stable to partition the data across a few chunks.
This allows it to fit in the repo comfortably and improve processing performance, which becomes relevant later. `bot_tweets` contains the ones flagged as Russian trolls. The rest are so called "normal" tweets.

At the end of this step, we can use `spacy`'s `vector` function to get a `word2vec` representation of each token in the string. Those are then saved to a file

#### **Warning!**: The matrix files are roughly 700 MB each, 10 partitions, per chunk of the data (there are 8 in total, including the bots file)

Our input layer will then accept an array of size 70 (a rough guesstimate of the highest token count) with each term containing the  

