import numpy as np

from sklearn import datasets
from sklearn.model_selection import train_test_split

from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.utils import np_utils

def build_logistic_regression():
    model = Sequential()
    model.add(Dense(3, input_shape=(4, )))
    model.add(Activation('softmax'))
    return model

def build_multilayer_perceptron():
    model = Sequential()
    model.add(Dense(16, input_shape=(4, )))
    model.add(Activation('relu'))
    model.add(Dense(3))
    model.add(Activation('softmax'))
    return model

if __name__ == "__main__":
    # load iris data
    iris = datasets.load_iris()
    X = iris.data
    Y = iris.target

    # one-hot-encoding label
    # 0 => [1, 0, 0]
    # 1 => [0, 1, 0]
    # 2 => [0, 0, 1]
    Y = np_utils.to_categorical(Y)

    # split the data
    train_X, test_X, train_Y, test_Y = train_test_split(X, Y, train_size=0.5)
    print(train_X.shape, test_X.shape, train_Y.shape, test_Y.shape)

    # logisitc regression or multilayer perceptron
    model = build_logistic_regression()
#    model = build_multilayer_perceptron()
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(train_X, train_Y, nb_epoch=500, batch_size=1, verbose=1)
    loss, accuracy = model.evaluate(test_X, test_Y, verbose=0)
    print("Accuracy = {:.2f}".format(accuracy))
