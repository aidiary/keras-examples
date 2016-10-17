import os
import numpy as np
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers.core import Dense, Activation

"""2-class logistic regression by keras"""

def plot_data(X, y):
    positive = [i for i in range(len(y)) if y[i] == 1]
    negative = [i for i in range(len(y)) if y[i] == 0]

    plt.scatter(X[positive, 0], X[positive, 1], c='red', marker='o', label='positive')
    plt.scatter(X[negative, 0], X[negative, 1], c='blue', marker='o', label='negative')

if __name__ == '__main__':
    # fix random seed
    seed = 1
    np.random.seed(seed)

    # load training data
    data = np.genfromtxt(os.path.join('data', 'ex2data1.txt'), delimiter=',')
    X = data[:, (0, 1)]
    y = data[:, 2]

    # plot training data
    plt.figure(1)
    plot_data(X, y)

    # create the model
    model = Sequential()
    model.add(Dense(1, input_shape=(2, )))
    model.add(Activation('sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # fit the model
    model.fit(X, y, nb_epoch=1000, batch_size=1, verbose=1)

    # get the learned weight
    weights = model.layers[0].get_weights()
    w1 = weights[0][0, 0]
    w2 = weights[0][1, 0]
    b = weights[1][0]

    # draw decision boundary
    plt.figure(1)
    xmin, xmax = min(X[:, 1]), max(X[:, 1])
    xs = np.linspace(xmin, xmax, 100)
    ys = [- (w1 / w2) * x - (b / w2) for x in xs]
    plt.plot(xs, ys, 'b-', label='decision boundary')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.xlim((30, 100))
    plt.ylim((30, 100))
    plt.legend()
    plt.show()
