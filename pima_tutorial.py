import os
import numpy as np
from keras.models import Sequential
from keras.layers import Dense

# http://machinelearningmastery.com/tutorial-first-neural-network-python-keras/

seed = 7
np.random.seed(seed)

if __name__ == "__main__":
    # load pima indians dataset
    dataset = np.loadtxt(os.path.join("data", "pima-indians-diabetes.data"), delimiter=',')

    # split into input (X) and output (Y) variables
    X = dataset[:, 0:8]
    Y = dataset[:, 8]

    # create model
    model = Sequential()
    model.add(Dense(12, input_dim=8, init='uniform', activation='relu'))
    model.add(Dense(8, init='uniform', activation='relu'))
    model.add(Dense(1, init='uniform', activation='sigmoid'))

    # compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    # fit the model
    model.fit(X, Y, nb_epoch=150, batch_size=10)

    # evaluate the model
    scores = model.evaluate(X, Y)
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))

    # prediction
    # round : probability => class
    predictions = np.round(model.predict(X))
    correct = Y[:, np.newaxis]

    np.set_printoptions(threshold=np.inf)
    print(np.hstack((predictions, correct)))
