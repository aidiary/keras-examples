import numpy as np
from keras.layers import Input, Dense, Convolution2D, MaxPooling2D, UpSampling2D
from keras.models import Model
from keras.datasets import mnist
import matplotlib.pyplot as plt

# https://blog.keras.io/building-autoencoders-in-keras.html

if __name__ == "__main__":
    # create model
    # use image_dim_ordering = th (~/.keras/keras.json)
    input_img = Input(shape=(1, 28, 28))

    x = Convolution2D(16, 3, 3, activation='relu', border_mode='same')(input_img)
    x = MaxPooling2D((2, 2), border_mode='same')(x)
    x = Convolution2D(8, 3, 3, activation='relu', border_mode='same')(x)
    x = MaxPooling2D((2, 2), border_mode='same')(x)
    x = Convolution2D(8, 3, 3, activation='relu', border_mode='same')(x)
    encoded = MaxPooling2D((2, 2), border_mode='same')(x)

    # at this point the representation is (8, 4, 4)

    x = Convolution2D(8, 3, 3, activation='relu', border_mode='same')(encoded)
    x = UpSampling2D((2, 2))(x)
    x = Convolution2D(8, 3, 3, activation='relu', border_mode='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Convolution2D(16, 3, 3, activation='relu')(x)
    x = UpSampling2D((2, 2))(x)
    decoded = Convolution2D(1, 3, 3, activation='sigmoid', border_mode='same')(x)

    autoencoder = Model(input_img, decoded)
    from keras.utils.visualize_util import plot
    plot(autoencoder, to_file='architecture.png', show_shapes=True)
    
    # compile model
    autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

    # load mnist datasets
    (X_train, _), (X_test, _) = mnist.load_data()

    X_train = X_train.astype('float32') / 255.0
    X_test = X_test.astype('float32') / 255.0

    X_train = np.reshape(X_train, (len(X_train), 1, 28, 28))
    X_test = np.reshape(X_test, (len(X_test), 1, 28, 28))

    print(X_train.shape)
    print(X_test.shape)

    # train model
    autoencoder.fit(X_train, X_train,
                    nb_epoch=50,
                    batch_size=128,
                    shuffle=True,
                    validation_data=(X_test, X_test))

    # encode and decode some digits
    decoded_imgs = autoencoder.predict(X_test)

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
