from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten
import numpy as np
import time
import FileReader
import tensorflow as tf
from sklearn.metrics import confusion_matrix
from keras.utils import to_categorical
import sklearn.metrics as metrics


def execute_cnn():
    # Model Template
    (x, x_test, x_train, x_cv, y, y_test, y_train, y_cv, all_tweets, all_labels) = FileReader.process()
    y = to_categorical(y)
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)
    print(all_tweets[0])
    print(len(all_tweets[0]))
    for val in all_tweets:
        print(len(val))

    model = Sequential() # declare model
    model.add(Dense(10, input_shape=(21000, ), kernel_initializer='he_normal')) # first layer
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
    model.compile(optimizer='sgd',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    # Train Model
    history = model.fit(x_train, y_train,
                        validation_data = (x_test, y_test),
                        epochs=1,
                        batch_size=512,
                        verbose=1)
    # Report Results
    predicted = model.predict(x_cv)
    n_values = 10
    y_pred = np.eye(n_values, dtype=int)[np.argmax(predicted, axis=1)]
    confusion = confusion_matrix(y_cv.argmax(axis=1), y_pred.argmax(axis=1))
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
    execute_cnn()