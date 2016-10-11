import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.utils import np_utils

# http://machinelearningmastery.com/understanding-stateful-lstm-recurrent-neural-networks-python-keras/

if __name__ == "__main__":
    # fix random seed for reproducibility
    np.random.seed(7)

    # define the raw dataset
    alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

    # create mapping of characters to integers (0-25) and the reverse
    char_to_int = dict((c, i) for i, c in enumerate(alphabet))
    int_to_char = dict((i, c) for i, c in enumerate(alphabet))

    # prepare the dataset of input to output pairs encoded as integers
    seq_length = 1
    dataX = []
    dataY = []
    for i in range(0, len(alphabet) - seq_length, 1):
        seq_in = alphabet[i:i + seq_length]
        seq_out = alphabet[i + seq_length]
        dataX.append([char_to_int[char] for char in seq_in])
        dataY.append(char_to_int[seq_out])

    # reshape X to be [samples, sequence length, features]
    X = np.reshape(dataX, (len(dataX), seq_length, 1))

    # normalize
    X = X / float(len(alphabet))

    # one hot encode the output variable
    y = np_utils.to_categorical(dataY)

    # create LSTM model
    batch_size = 1
    model = Sequential()
    model.add(LSTM(32, batch_input_shape=(batch_size, X.shape[1], X.shape[2]), stateful=True))
    model.add(Dense(y.shape[1], activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # fit the model
    for i in range(300):
        model.fit(X, y, nb_epoch=1, batch_size=batch_size, verbose=2, shuffle=False)
        model.reset_states()

    # summarize performance of the model
    scores = model.evaluate(X, y, batch_size=batch_size, verbose=0)
    model.reset_states()
    print("Model accuracy (train data): %.2f%%" % (scores[1] * 100))

    # demonstrate some model predictions
    seed = [char_to_int[alphabet[0]]]
    for i in range(0, len(alphabet) - 1):
        x = np.reshape(seed, (1, len(seed), 1))
        x = x / float(len(alphabet))
        prediction = model.predict(x, verbose=0)
        index = np.argmax(prediction)
        print(int_to_char[seed[0]], '->', int_to_char[index])
        seed = [index]
    model.reset_states()

    # demonstrate a random starting point
    letter = 'K'
    seed = [char_to_int[letter]]
    print("New start:", letter)
    for i in range(0, 5):
        x = np.reshape(seed, (1, len(seed), 1))
        x = x / float(len(alphabet))
        prediction = model.predict(x, verbose=0)
        index = np.argmax(prediction)
        print(int_to_char[seed[0]], '->', int_to_char[index])
    model.reset_states()
