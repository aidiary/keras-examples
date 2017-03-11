from keras.models import Sequential
from keras.layers import Dense, Activation, Reshape, Dropout, Flatten
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import UpSampling2D, Convolution2D
from keras.layers.advanced_activations import LeakyReLU
from keras.datasets import mnist
from keras.optimizers import Adam

import os
import math
import numpy as np
from PIL import Image


# https://elix-tech.github.io/ja/2017/02/06/gan.html


def generator_model():
    model = Sequential()
    model.add(Dense(input_dim=100, output_dim=1024))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dense(128 * 7 * 7))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Reshape((128, 7, 7), input_shape=(128 * 7 * 7,)))
    model.add(UpSampling2D((2, 2)))  # 14x14
    model.add(Convolution2D(64, 5, 5, border_mode='same'))
    model.add(Activation('tanh'))
    model.add(UpSampling2D(size=(2, 2)))  # 28x28
    model.add(Convolution2D(1, 5, 5, border_mode='same'))
    model.add(Activation('tanh'))
    return model


def descriminator_model():
    model = Sequential()
    model.add(Convolution2D(64, 5, 5,
        subsample=(2, 2),
        border_mode='same',
        input_shape=(28, 28, 1)))
    model.add(LeakyReLU(0.2))
    model.add(Convolution2D(128, 5, 5, subsample=(2, 2)))
    model.add(LeakyReLU(0.2))
    model.add(Flatten())
    model.add(Dense(256))
    model.add(LeakyReLU(0.2))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    return model


def combine_images(generated_images):
    total = generated_images.shape[0]
    cols = int(math.sqrt(total))
    rows = math.ceil(float(total) / cols)
    width, height = generated_images.shape[2:]
    combined_image = np.zeros((height * rows, width * cols), dtype=generated_images.dtype)

    for index, image in enumerate(generated_images):
        i = int(index / cols)
        j = index % cols
        combined_image[width * i:width * (i + 1), height * j:height * (j + 1)] = image[0, :, :]
    return combined_image


BATCH_SIZE = 32
NUM_EPOCH = 20
GENERATED_IMAGE_PATH = 'generated_images/'


def train():
    (X_train, y_train), (_, _) = mnist.load_data()

    # -1から1の範囲に
    X_train = (X_train.astype(np.float32) - 127.5) / 127.5
    # 4Dテンソルへ (60000, 1, 28, 28)
    X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1], X_train.shape[2])

    print(X_train.shape)
    print(X_train.min(), X_train.max())

    descriminator = descriminator_model()
    d_opt = Adam(lr=1e-5, beta_1=0.1)
    descriminator.compile(loss='binary_crossentropy', optimizer=d_opt)

    descriminator.trainable = False
    generator = generator_model()

    descriminator.summary()
    generator.summary()

    # dcgan = Sequential([generator, descriminator])
    # g_opt = Adam(lr=2e-4, beta_1=0.5)
    # dcgan.compile(loss='binary_crossentropy', optimizer=g_opt)
    #
    # dcgan.summary()


if __name__ == '__main__':
    train()
