from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten
from keras.models import load_model
from keras.optimizers import SGD
import itertools as IT
import numpy as np
import time
import datetime
import os
import tensorflow as tf
from sklearn.metrics import confusion_matrix
from keras.utils import to_categorical
import sklearn.metrics as metrics
from constants import MAX_VECTOR_COUNT, LEARNING_RATE, CHUNKS_PER_ITERATION, PASSES
import global_processor
import random
import logging
import sys

root = logging.getLogger()
root.setLevel(logging.DEBUG)
ch = logging.StreamHandler(sys.stdout)
ch.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
root.addHandler(ch)

def execute_network(x_train, x_test, y_train, y_test, model_name):
    # Model Template
    logging.info("Initializing model using naive file processing.")
    y_train = to_categorical(y_train, num_classes=2)
    y_test = to_categorical(y_test, num_classes=2)
    logging.info("Data received from FileReader. Converting label data to categorical format...")

    logging.info("Final prep complete, model initializing")
    model = Sequential()  # declare model
    model.add(Dense(1000, input_shape=(MAX_VECTOR_COUNT, ), kernel_initializer='he_normal'))  # first layer
    model.add(Activation('linear'))
    model.add(Dense(2000, activation='linear'))
    model.add(Dense(2000, activation='linear'))
    model.add(Dense(2, kernel_initializer='uniform'))  # last layer
    model.add(Activation('sigmoid'))
    model.summary()
    # Compile Model
    logging.info("Compiling model...")
    optimizer = SGD(lr=LEARNING_RATE)

    model.compile(optimizer='sgd',
                  loss='binary_crossentropy',
                  metrics=['accuracy', 'binary_accuracy'])
    logging.info("Model compilation complete.")
    # Train Model
    history = model.fit(x_train, y_train,
                        validation_data=(x_test, y_test),
                        epochs=2,
                        batch_size=2048,
                        verbose=1)
    logging.info("Epochs complete. Saving model...")
    model.save(model_name)
    logging.info("Model saved. Use predict_on_model() to see cross validation results.")
    return model


def predict_on_model(data_path, label_path, model_name):
    logging.info("Getting cross validation set from FileReader...")
    x_cv, y_cv = global_processor.process_no_split(data_path, label_path)
    logging.info("Loading model...")
    model = load_model(model_name)
    logging.info("Load complete.")
    # Report Results
    predicted = model.predict(x_cv)
    y_pred = []
    for val in predicted:
        logging.info("Predicted {0}".format(val))
        if(val[0] > val[1]):
            y_pred.append(0)
        else:
            y_pred.append(1)
    y_pred = np.array(y_pred)
    confusion = confusion_matrix(y_cv, y_pred)
    acc = metrics.accuracy_score(y_cv, y_pred)
    prec = metrics.precision_score(y_cv, y_pred, average=None)
    rec = metrics.recall_score(y_cv, y_pred, average=None)
    print(confusion)
    print(acc)
    print(rec)
    print(prec)


def train_existing_model(model, x_train, x_test, y_train, y_test, model_name, trainCount):
    logging.info("Initializing model using naive file processing.")
    logging.info("Data received. Converting label data to categorical format...")
    y_train = to_categorical(y_train, num_classes=2)
    y_test = to_categorical(y_test, num_classes=2)
    logging.info("Done.")
    history = model.fit(x_train, y_train,
                        validation_data=(x_test, y_test),
                        epochs=2,
                        batch_size=2048,
                        verbose=1)
    logging.info("Epochs complete. Saving model...")
    new_model_name = model_name.split("_")[0] + "_" + str(trainCount)
    logging.info("Model saved. Use predict_on_model() to see cross validation results.")
    return [model, new_model_name]

def evenly_spaced(*iterables):
    """
    >>> evenly_spaced(range(10), list('abc'))
    [0, 1, 'a', 2, 3, 4, 'b', 5, 6, 7, 'c', 8, 9]
    """
    return [item[1] for item in
            sorted(IT.chain.from_iterable(
            zip(IT.count(start=1.0 / (len(seq) + 1),
                         step=1.0 / (len(seq) + 1)), seq)
            for seq in iterables))]

def train_network_on_all_data():
    # base_name, chunk_lower, chunk_higher, fileNo
    first = True
    model = None
    last_model_name = "model_0"
    training_iteration_count = 0
    data_dir = input("Input data dir: ")
    files = os.listdir(data_dir)
    bot_files = [f for f in files if "bot" in f]
    regular_files = [f for f in files if "regular" in f]
    random.shuffle(bot_files)
    random.shuffle(regular_files)
    len_both = len(bot_files) + len(regular_files)
    a = [f for f in files if "bot" in f]
    b = [f for f in files if "regular" in f]
    logging.info(a)
    logging.info(b)
    interleaved_file_list = evenly_spaced(a, b)
    logging.info(interleaved_file_list)


    input()
    x_train = None
    x_test = None
    y_train = None
    y_test = None
    loaded_count = 0  # loads CHUNKS_PER_ITERATION chunks
    steps_to_save = 0
    for iteration in range(0, PASSES):
        for file in files:
            if x_train is None:
                x_train, x_test, y_train, y_test = global_processor.load_chunk(os.path.join(data_dir, file))
            else:
                loaded_x_train, loaded_x_test, loaded_y_train, loaded_y_test = global_processor.load_chunk(
                    os.path.join(data_dir, file))
                x_train = np.concatenate([x_train, loaded_x_train])
                x_test = np.concatenate([x_test, loaded_x_test])
                y_train = np.concatenate([y_train, loaded_y_train])
                y_test = np.concatenate([y_test, loaded_y_test])
            loaded_count += 1
            if loaded_count >= CHUNKS_PER_ITERATION:
                if first:
                    model = execute_network(
                        x_train,
                        x_test,
                        y_train,
                        y_test,
                        last_model_name)
                    first = False
                else:
                    [model, last_model_name] = train_existing_model(
                        model,
                        x_train,
                        x_test,
                        y_train,
                        y_test,
                        last_model_name,
                        training_iteration_count)
                    training_iteration_count += 1
                # reset arrs
                loaded_count = 0
                steps_to_save += 1
                x_train = None
                x_test = None
                y_train = None
                y_test = None
        logging.info("Saving model with name {0}.".format(last_model_name))
        model.save(last_model_name)
    logging.info("All processing complete! Completed " + str(training_iteration_count) + " iterations.")


if __name__ == "__main__":
    train_network_on_all_data()
