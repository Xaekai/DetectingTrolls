from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten
import numpy as np
import time
import FileReader
import tensorflow as tf
from sklearn.metrics import confusion_matrix
from keras.utils import to_categorical
import sklearn.metrics as metrics


def execute_cnn_no_generator():
    # Model Template
    print("Initializing model using naive file processing.")
    (x_test, x_train, x_cv, y_test, y_train, y_cv) = FileReader.process()
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)
    print("Data received from FileReader. Converting label data to categorical format...")

    print("Final prep complete, model initializing")
    model = Sequential() # declare model
    model.add(Dense(1000, input_shape=(40*300, ), kernel_initializer='he_normal')) # first layer
    model.add(Activation('relu'))
    model.add(Dense(5096, activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(5096, activation='relu'))
    model.add(Dense(5096, activation='sigmoid'))
    model.add(Dense(5096, activation='relu'))
    model.add(Dense(2, kernel_initializer='he_normal')) # last layer
    model.add(Activation('softmax'))
    model.summary()
    # Compile Model
    print("Compiling model...")
    model.compile(optimizer='sgd',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    print("Model compilation complete.")
    # Train Model
    history = model.fit(x_train, y_train,
                        validation_data = (x_test, y_test),
                        epochs=100,
                        batch_size=512,
                        verbose=1)
    # Report Results
    predicted = model.predict(x_cv)
    model.save("troll_model")
    n_values = 1
    y_pred = np.eye(n_values, dtype=int)[np.argmax(predicted)]
    confusion = confusion_matrix(y_cv.argmax(), y_pred)
    acc = metrics.accuracy_score(y_cv, y_pred)
    prec = metrics.precision_score(y_cv, y_pred, average=None)
    rec = metrics.recall_score(y_cv, y_pred, average=None)
    whoops = [] #figure out which ones were misclassified, dubbed the "whoops array"
    for val in range(0, len(y_pred)):
        if(np.array_equal(y_cv[val], y_pred[val])):
            pass
        else:
            whoops.append((y_pred[val], x[val]))

    print(confusion)
    print(acc)
    print(rec)
    print(prec)

def execute_cnn_generator():
    # Model Template
    print("Initializing model using a generator.")

    print("Model initializing")
    model = Sequential() # declare model
    model.add(Dense(1000, input_shape=(40*300, ), kernel_initializer='he_normal')) # first layer
    model.add(Activation('relu'))
    model.add(Dense(5096, activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(5096, activation='relu'))
    model.add(Dense(5096, activation='sigmoid'))
    model.add(Dense(5096, activation='relu'))
    model.add(Dense(2, kernel_initializer='he_normal')) # last layer
    model.add(Activation('softmax'))
    model.summary()
    # Compile Model
    print("Compiling model...")
    model.compile(optimizer='sgd',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    print("Model compilation complete.")
    # Train Model
    history = model.fit_generator(FileReader.load_raveled_generator(0, 5, 0, 8), validation_data=FileReader.load_raveled_generator(5, 7, 0, 8), validation_steps=1000,
                        epochs=100,
                        steps_per_epoch=10000,
                        verbose=1)
    # Report Results
    predicted = model.predict(x_cv)
    model.save("troll_model")
    n_values = 1
    y_pred = np.eye(n_values, dtype=int)[np.argmax(predicted, axis=1)]
    confusion = confusion_matrix(y_cv.argmax(axis=1), y_pred)
    acc = metrics.accuracy_score(y_cv, y_pred)
    prec = metrics.precision_score(y_cv, y_pred, average=None)
    rec = metrics.recall_score(y_cv, y_pred, average=None)
    whoops = [] #figure out which ones were misclassified, dubbed the "whoops array"
    for val in range(0, len(y_pred)):
        if(np.array_equal(y_cv[val], y_pred[val])):
            pass
        else:
            whoops.append((y_pred[val], x[val]))

    print(confusion)
    print(acc)
    print(rec)
    print(prec)

if __name__ == "__main__":
    execute_cnn_generator()