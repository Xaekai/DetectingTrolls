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
from constants import MAX_VECTOR_COUNT
import FileReader

def execute_cnn_no_generator():
    # Model Template
    print("Initializing model using naive file processing.")
    (x_test, x_train, x_cv, y_test, y_train, y_cv) = FileReader.process()
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)
    print("Data received from FileReader. Converting label data to categorical format...")

    print("Final prep complete, model initializing")
    model = Sequential() # declare model
    model.add(Dense(1000, input_shape=(MAX_VECTOR_COUNT, ), kernel_initializer='he_normal')) # first layer
    model.add(Activation('linear'))
    model.add(Dense(5096, activation='linear'))
    model.add(Dense(5096, activation='softplus'))
    model.add(Dense(5096, activation='sigmoid'))
    model.add(Dense(5096, activation='linear'))
    model.add(Dense(2, kernel_initializer='he_normal')) # last layer
    model.add(Activation('softmax'))
    model.summary()
    # Compile Model
    print("Compiling model...")
    optimizer = SGD(lr=0.001)

    model.compile(optimizer='sgd',
                  loss='binary_crossentropy',
                  metrics=['accuracy', 'binary_accuracy'])
    print("Model compilation complete.")
    # Train Model
    history = model.fit(x_train, y_train,
                        validation_data = (x_test, y_test),
                        epochs=1,
                        batch_size=2048,
                        verbose=1)
    print("Epochs complete. Saving model...")
    model.save("troll_model")
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

def train_existing_model():
    print("Initializing model using naive file processing.")
    (x_test, x_train, x_cv, y_test, y_train, y_cv) = FileReader.process()
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)
    print("Data received from FileReader. Converting label data to categorical format...")
    print("Loading model...")
    model = load_model("troll_model")
    print("Model loaded.")
    history = model.fit(x_train, y_train,
                        validation_data = (x_test, y_test),
                        epochs=1,
                        batch_size=2048,
                        verbose=1)
    print("Epochs complete. Saving model...")
    model.save("troll_model_trained")
    print("Model saved. Use predict_on_model() to see cross validation results.")
if __name__ == "__main__":
    execute_cnn_no_generator()
    predict_on_model()