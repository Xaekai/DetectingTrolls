from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten
from keras.models import load_model
from keras.optimizers import SGD
import spacy
import csv
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.externals import joblib
import time
import datetime
import os
import tensorflow as tf
from sklearn.metrics import confusion_matrix
from keras.utils import to_categorical
import sklearn.metrics as metrics
from constants import MAX_VECTOR_COUNT, LEARNING_RATE, CHUNKS_PER_ITERATION, PASSES
import global_processor
from common import *


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
    return model


def predict_on_model(model, x_cv, y_cv):
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
    report = metrics.classification_report(y_cv, y_pred)
    logging.info("Confusion matrix: {0}".format(confusion))
    logging.info("Accuracy {0}".format(acc))
    logging.info("Recall {0}".format(rec))
    logging.info("Precision {0}".format(prec))
    logging.info("CLASSIFICATION REPORT\n{0}".format(report))


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


def load_validation_data(dir, model_path):
    x_train = None
    x_test = None
    y_train = None
    y_test = None
    interleaved_file_list = get_interleaved_file_list(dir)
    for file in interleaved_file_list:
        if x_train is None:
            x_train, x_test, y_train, y_test = global_processor.load_chunk(os.path.join(dir, file))
        else:
            loaded_x_train, loaded_x_test, loaded_y_train, loaded_y_test = global_processor.load_chunk(
                os.path.join(dir, file))
            x_train = np.concatenate([x_train, loaded_x_train])
            x_test = np.concatenate([x_test, loaded_x_test])
            y_train = np.concatenate([y_train, loaded_y_train])
            y_test = np.concatenate([y_test, loaded_y_test])
    model = load_model(model_path)
    y_all = np.concatenate([y_train, y_test])
    x_all = np.concatenate([x_train, x_test])
    logging.info("Available samples {0}".format(x_all.size))
    predict_on_model(model, x_all, y_all)


def validate_on_text_data(data_dir, model_name):
    interleaved_file_list = get_interleaved_file_list(data_dir)
    logging.info("Loading Spacy's large vector set. ")
    nlp = spacy.load('en_vectors_web_lg')
    logging.info("Finished loading. ")
    for file in interleaved_file_list:
        file_path = os.path.join(data_dir, file)
        is_bot = is_bot_file(file_path)
        with open(file_path, 'r', encoding="utf8") as tweet_file:
            reader = csv.DictReader(tweet_file)
            tweets = []
            for line in reader:
                text = line['text']
                no_emojis = filter_emojis(text)
                rowTokens = []
                tokens = nlp(no_emojis)
                for tok in tokens:
                    rowTokens.append(tok.vector)
                tweet_filtered = np.concatenate(rowTokens).ravel()
                tweets.append(tweet_filtered)
            tweets_padded = pad_tweet_arr(tweets)
            scaler = joblib.load('scaler.pkl')
            all_scaled = scaler.transform(tweets_padded)
            labels = []
            if is_bot:
                labels = np.ones(len(all_scaled), dtype=int)
            else:
                labels = np.zeros(len(all_scaled), dtype=int)
            model = load_model(model_name)
            predict_on_model(model, all_scaled, labels)


def rebuild_scaler(dir):
    interleaved_file_list = get_interleaved_file_list(dir)
    scaler = StandardScaler()
    for file in interleaved_file_list:
        logging.info("Loading file for scaling {0}".format(file))
        matrix = global_processor.load_raw(os.path.join(dir, file))
        scaler.partial_fit(matrix)
        logging.info("Partial fit on matrix file complete.")
    joblib.dump(scaler, 'scaler.pkl')


def train_network_on_all_data(data_dir):
    # base_name, chunk_lower, chunk_higher, fileNo
    first = True
    model = None
    last_model_name = "model_0"
    training_iteration_count = 0

    x_train = None
    x_test = None
    y_train = None
    y_test = None
    loaded_count = 0  # loads CHUNKS_PER_ITERATION chunks
    interleaved_file_list = get_interleaved_file_list(data_dir)
    for file in interleaved_file_list:
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
            logging.info("Saving model with name {0}.".format(last_model_name))
            model.save(last_model_name)
            x_train = None
            x_test = None
            y_train = None
            y_test = None

    logging.info("All processing complete! Completed " + str(training_iteration_count) + " iterations.")


def menu():
    """generic menu harness"""
    while True:
        print("1: Train network on folder")
        print("2: Validate on data")
        print("3: Validate on text data")
        print("4: Rebuild scaler on data")
        print("5: Exit")
        choice = input("Select option: ")
        if choice == "1":
            data_dir = input("Input data dir: ")
            train_network_on_all_data(data_dir)
        elif choice == "2":
            model_path = input("Enter path to model: ")
            validation_data_dir = input("Enter path to validation data directory: ")
            load_validation_data(validation_data_dir, model_path)
        elif choice == "3":
            model_path = input("Enter path to model: ")
            validation_data_dir = input("Enter path to validation data directory: ")
            validate_on_text_data(validation_data_dir, model_path)
        elif choice == "4":
            data_dir = input("Input data dir: ")
            rebuild_scaler(data_dir)
        elif choice == "5":
            exit(0)


if __name__ == "__main__":
    menu()
