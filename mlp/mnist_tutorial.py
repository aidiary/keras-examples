import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (7, 7)

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.utils import np_utils

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

if __name__ == "__main__":
    nb_classes = 10

    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    print("X_train original shape", X_train.shape)
    print("y_train original shape", y_train.shape)

    for i in range(9):
        plt.subplot(3, 3, i + 1)
        plt.imshow(X_train[i], cmap='gray', interpolation='none')
        plt.title("Class {}".format(y_train[i]))
    plt.show()

    X_train = X_train.reshape(60000, 784).astype('float32')
    X_test = X_test.reshape(10000, 784).astype('float32')
    X_train /= 255
    X_test /= 255
    print("Training matrix shape", X_train.shape)
    print("Testing matrix shape", X_test.shape)

    y_train_ohe = np_utils.to_categorical(y_train, nb_classes)
    y_test_ohe = np_utils.to_categorical(y_test, nb_classes)
    print("y_train_ohe shape", y_train_ohe.shape)

    model = Sequential()

    model.add(Dense(512, input_shape=(784, )))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))

    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))

    model.add(Dense(10))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    model.fit(X_train, y_train_ohe, batch_size=128, nb_epoch=4,
              verbose=1, validation_data=(X_test, y_test_ohe))

    score = model.evaluate(X_test, y_test_ohe, verbose=1)
    print("Test score: ", score[0]);
    print("Test accuracy: ", score[1]);

    predicted_classes = model.predict_classes(X_test)

    correct_indices = np.nonzero(predicted_classes == y_test)[0]
    incorrect_indices = np.nonzero(predicted_classes != y_test)[0]

    plt.figure()
    for i, correct in enumerate(correct_indices[:9]):
        plt.subplot(3, 3, i + 1)
        plt.imshow(X_test[correct].reshape(28, 28), cmap='gray', interpolation='none')
        plt.title("Predicted {}, Class {}".format(predicted_classes[correct], y_test[correct]))

    plt.figure()
    for i, incorrect in enumerate(incorrect_indices[:9]):
        plt.subplot(3, 3, i + 1)
        plt.imshow(X_test[incorrect].reshape(28, 28), cmap='gray', interpolation='none')
        plt.title("Predicted {}, Class {}".format(predicted_classes[incorrect], y_test[incorrect]))

    plt.show()

