# Classifying Russian Troll Tweets via Artificial Neural Nets

## Data
NBC Universal has provided 200,000 tweets determined to be related in some way to Russian trolls. You can find the dataset [here](nbcnews.com/tech/social-media/now-available-more-200-000-deleted-russian-troll-tweets-n844731?cid=sm_npd_nn_tw_ma)

I combined this with a dataset of 1.6m "regular" tweets located [here](http://thinknook.com/twitter-sentiment-analysis-training-corpus-dataset-2012-09-22/). Originally intended for sentiment analysis, but it provides a great base for "regular" tweets.

## Goal
The hope is that we will be able to classify tweets as Russian troll tweets given this training set. I don't expect to see anything even close to 80% accuracy, but if it can help find the needle in the haystack, it's a net positive.
It's also a nice educational exercise in data wrangling and neural nets.

## Implementation

### Preprocessing / Wrangling

This step parses existing data using the `csv` library in Python, `numpy` and a few other tools. Most notably, it **strips out the emojis**. This is important because otherwise when we attempt to perform NLP on it, 
there will be no useful vectors for the emojis. Plus, emojis seem to consistently break a variety of Python tools and I'd rather not deal with it. 

I used AlexDel's [csv_splitter](https://gist.github.com/AlexDel) and made a few modifications to make it more stable to partition the data across a few chunks.
This allows it to fit in the repo comfortably and improve processing performance, which becomes relevant later. `bot_tweets` contains the ones flagged as Russian trolls. The rest are so called "normal" tweets.

Here, `SpaCy` and `word2vec` become crucial. word2vec s the algorithm described in the seminal paper by Google [here](https://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality.pdf)

At the end of this step, we can use `spacy`'s `vector` function to get a `word2vec` representation of each token in the string. The vectors are raveled such that one tweet is a single row of vectors in unrolled in order. Those are then saved to a file in 10 chunks.

**Warning!**: The matrix files are roughly 700 MB each, 10 partitions, per chunk of the data (there are 8 in total, including the bots file)


### File processing

The file processor then reads in four chunks of data, 75% "regular" and 25% "bot". This is a rough approximation of the distribution of the original data set. Due to the extreme amounts of memory required to do this in Python, this will serve as our stratified sampling.
Those chunks are padded with zeroes or trimmed such that each tweet has at most 40 tokens. Observations during training show that only a tiny portion (100-200 per 100,000 tweets) have more than 40 tokens. A sparse matrix might be more appropriate here, but I don't have the expertise for that yet.

Once this chunk is prepared, it is saved. It can then be read in by the network later on.

This file is distinct from the preprocessor for a few reasons.
1) This procesing was done in many steps. All the steps in one file would be a lot to sort through.
2) The preprocessor does the heavy lifting for NLP. The SpaCy vector set is large, and loading the matrices and the NLP at the same time uses way too much RAM.
3) The `global_processor` does the heavy lifting from the vectorized format onwards. It performs global alterations to the data, such as raveling, padding, standardization, stratification, and shuffling.

### Network

When a chunk is loaded, we use `scikit_learn`'s `StandardScaler` to transform the data to be better formatted for the network.
The network trains on it for two epochs with a batch size of 2048 (this is also a rough guess, each chunk is roughly 64k tweets for training).

Our input layer will then accept an array of size 12,000 (`word2vec` provides a size 300 matrix for every token, trimmed to 40 tokens) (a rough guesstimate of the highest token count) with each row containing a tweet.
We then apply two dense linear layers with 2,000 neurons each. This was tuned via trial and error. Dropout, softmax and sigmoid tends to interfere here. Our output layer is a sigmoid layer that accepts one hot encodings of the input labels.
The one hot encoding could probably be eliminated, but it provides a great deal of flexibility if future classes are added.

### Optimizer / loss function
Standard stochastic gradient descent (SGD) with a reduced learning rate is the choice here. The search space is massive, the default 0.01 learning rate leads to local maxima. A few others proved ineffective in testing.
For the loss function, binary crossentropy is the obvious choice as there are only two classes.

### Training process

The network trains on chunks four at a time. This is mostly due to the massive memory demands. After two epochs, the model is saved, then reloaded, and training resumes. A new file will be created for each step in the model.


## System requirements
Working with huge matrices and large amounts of file IO is demanding. **Your system will crash if it runs 32 bit Python. There is not enough room in the address space to load even a single chunk.**
Keras is backed with Tensorflow, so it will use your GPU while running.
**Expected RAM requirements: 6gb with surges up to 10gb**

**Expected disk space requirements: 100+gb free is ideal here.**

## Future improvements

More data would be great, though I'm not sure where to go get it. I will try to test on completely new tweets as I'm able to find them. Any new data found would be greatly appreciated! Better PEP8 compliance? A real interface? Deduplicate some `global_processor` code?
Less generic file names?