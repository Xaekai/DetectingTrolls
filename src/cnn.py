from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten
from keras.models import load_model
from keras.optimizers import SGD
import numpy as np
import time
import datetime

import tensorflow as tf
from sklearn.metrics import confusion_matrix
from keras.utils import to_categorical
import sklearn.metrics as metrics
from constants import MAX_VECTOR_COUNT, LEARNING_RATE
import FileReader

def execute_network(data_path, label_path, model_name):
    # Model Template
    print("Initializing model using naive file processing.")
    x_train, x_test, y_train, y_test = FileReader.process(data_path, label_path)
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)
    print("Data received from FileReader. Converting label data to categorical format...")

    print("Final prep complete, model initializing")
    model = Sequential()  # declare model
    model.add(Dense(1000, input_shape=(MAX_VECTOR_COUNT, ), kernel_initializer='he_normal'))  # first layer
    model.add(Activation('linear'))
    model.add(Dense(2000, activation='linear'))
    model.add(Dense(2000, activation='linear'))
    model.add(Dense(2, kernel_initializer='uniform'))  # last layer
    model.add(Activation('sigmoid'))
    model.summary()
    # Compile Model
    print("Compiling model...")
    optimizer = SGD(lr=LEARNING_RATE)

    model.compile(optimizer='sgd',
                  loss='binary_crossentropy',
                  metrics=['accuracy', 'binary_accuracy'])
    print("Model compilation complete.")
    # Train Model
    history = model.fit(x_train, y_train,
                        validation_data = (x_test, y_test),
                        epochs=2,
                        batch_size=2048,
                        verbose=1)
    print("Epochs complete. Saving model...")
    model.save(model_name)
    print("Model saved. Use predict_on_model() to see cross validation results.")

def predict_on_model():
    print("Getting cross validation set from FileReader...")
    x_cv, y_cv = FileReader.get_cv_set()
    print("Loading model...")
    model = load_model("troll_model")
    print("Load complete.")
    # Report Results
    predicted = model.predict(x_cv)
    y_pred = []
    for val in predicted:
        print(val)
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

def train_existing_model(data_path, label_path, model_name, trainCount):
    print("Initializing model using naive file processing.")
    x_train, x_test, y_train, y_test = FileReader.process(data_path, label_path)
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)
    print("Data received from FileReader. Converting label data to categorical format...")
    print("Loading model...")
    model = load_model(model_name)
    print("Model loaded.")
    history = model.fit(x_train, y_train,
                        validation_data=(x_test, y_test),
                        epochs=2,
                        batch_size=2048,
                        verbose=1)
    print("Epochs complete. Saving model...")
    model.save(model_name + "_" + str(trainCount))
    print("Model saved. Use predict_on_model() to see cross validation results.")
    return model_name + "_" + str(trainCount)


if __name__ == "__main__":
    # base_name, chunk_lower, chunk_higher, fileNo
    first = True
    last_model_name = "troll_model"
    training_iteration_count = 0
    for fileNo in range(0, 6):
        for chunk_lower in range(0, 9, 3):
            base_dir = "../data/large_chunk_" + str(fileNo) + "_" + str(chunk_lower) + "_" + str(chunk_lower + 3)
            FileReader.build_large_arr(
                base_dir,
                fileNo + chunk_lower // 3,
                chunk_lower,
                chunk_lower + 3,
                fileNo)
            # We break up data into these chunks partially as an artifact of preprocessing
            # but more realistically because
            tweet_file = base_dir + "_tweets"
            label_file = base_dir + "_labels"
            if first:
                execute_network(tweet_file, label_file, last_model_name)
                first = False
            else:
                last_model_name = train_existing_model(tweet_file, label_file, last_model_name, training_iteration_count)
                training_iteration_count += 1
