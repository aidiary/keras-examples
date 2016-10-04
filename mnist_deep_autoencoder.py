import numpy as np
from keras.layers import Input, Dense
from keras.models import Model
from keras.datasets import mnist
import matplotlib.pyplot as plt

# https://blog.keras.io/building-autoencoders-in-keras.html

if __name__ == "__main__":
    # create model
    input_img = Input(shape=(784, ))
    encoded = Dense(128, activation='relu')(input_img)
    encoded = Dense(64, activation='relu')(encoded)
    encoded = Dense(32, activation='relu')(encoded)
    
    decoded = Dense(64, activation='relu')(encoded)
    decoded = Dense(128, activation='relu')(decoded)
    decoded = Dense(784, activation='sigmoid')(decoded)

    autoencoder = Model(input=input_img, output=decoded)

    # compile model
    autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

    # load mnist datasets
    (X_train, _), (X_test, _) = mnist.load_data()

    X_train = X_train.astype('float32') / 255.0
    X_test = X_test.astype('float32') / 255.0

    X_train = X_train.reshape((len(X_train), np.prod(X_train.shape[1:])))
    X_test = X_test.reshape((len(X_test), np.prod(X_test.shape[1:])))

    print(X_train.shape)
    print(X_test.shape)

    # train model
    autoencoder.fit(X_train, X_train,
                    nb_epoch=100,
                    batch_size=256,
                    shuffle=True,
                    validation_data=(X_test, X_test))

    # create a separate encoder model
    layers = autoencoder.layers
    encoder = Model(input=input_img,
                    output=encoded)

    # create a separate decoder model
    encoded_input = Input(shape=(32, ))
    layers = autoencoder.layers
    decoded_layers = layers[-1]
    decoder = Model(input=encoded_input,
                    output=layers[-1](layers[-2](layers[-3](encoded_input))))

    # encode and decode some digits
    encoded_imgs = encoder.predict(X_test)
    decoded_imgs = decoder.predict(encoded_imgs)

    n = 10
    plt.figure(figsize=(20, 4))
    for i in range(n):
        # display original
        ax = plt.subplot(2, n, i + 1)
        plt.imshow(X_test[i].reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # display reconstruction
        ax = plt.subplot(2, n, i + 1 + n)
        plt.imshow(decoded_imgs[i].reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()
