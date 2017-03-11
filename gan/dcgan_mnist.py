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
    model.add(Dense(7 * 7 * 128))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    # tfモードの場合はチャネルは後！
    model.add(Reshape((7, 7, 128), input_shape=(7 * 7 * 128,)))
    model.add(UpSampling2D((2, 2)))  # 画像のサイズが2倍になる 14x14
    model.add(Convolution2D(64, 5, 5, border_mode='same'))
    model.add(Activation('tanh'))
    model.add(UpSampling2D(size=(2, 2)))  # 28x28
    model.add(Convolution2D(1, 5, 5, border_mode='same'))  # 28x28x1が出力
    model.add(Activation('tanh'))
    return model


def discriminator_model():
    model = Sequential()
    # Poolingの代わりにsubsample使う（出力のサイズ半分）
    model.add(Convolution2D(64, 5, 5,
              subsample=(2, 2),
              border_mode='same',
              input_shape=(28, 28, 1)))  # tfモードはチャネルは後！
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
    width, height = generated_images.shape[1:3]
    combined_image = np.zeros((height * rows, width * cols), dtype=generated_images.dtype)

    for index, image in enumerate(generated_images):
        i = int(index / cols)
        j = index % cols
        combined_image[width*i:width*(i+1), height*j:height*(j+1)] = image[:, :, 0]
    return combined_image


BATCH_SIZE = 32
NUM_EPOCH = 20
GENERATED_IMAGE_PATH = 'generated_images/'


def train():
    (X_train, y_train), (_, _) = mnist.load_data()

    # -1から1の範囲に
    X_train = (X_train.astype(np.float32) - 127.5) / 127.5
    # 4Dテンソルへ (60000, 28, 28, 1)
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], 1)

    print(X_train.shape)
    print(X_train.min(), X_train.max())

    discriminator = discriminator_model()
    d_opt = Adam(lr=1e-5, beta_1=0.1)
    discriminator.compile(loss='binary_crossentropy', optimizer=d_opt)

    # dcgan内のdiscriminatorは重みを更新しない
    discriminator.trainable = False
    generator = generator_model()

    # discriminator.summary()
    # generator.summary()

    # Sequentialには層のリストが渡せる
    # generatorのあとにdiscriminatorが接続されるモデル
    # 入力 (100, ) => generator => (28, 28, 1) => discriminator => (1, )
    dcgan = Sequential([generator, discriminator])
    g_opt = Adam(lr=2e-4, beta_1=0.5)
    dcgan.compile(loss='binary_crossentropy', optimizer=g_opt)

    # dcgan.summary()

    num_batches = int(X_train.shape[0] / BATCH_SIZE)
    print('Number of batches:', num_batches)
    for epoch in range(NUM_EPOCH):
        for index in range(num_batches):
            # 入力はノイズ画像
            noise = np.array([np.random.uniform(-1, 1, 100) for _ in range(BATCH_SIZE)])
            image_batch = X_train[index * BATCH_SIZE:(index + 1) * BATCH_SIZE]
            generated_images = generator.predict(noise, verbose=0)  # 32x28x28x1

            # 500バッチごとに生成画像を出力
            if index % 500 == 0:
                image = combine_images(generated_images)
                image = image * 127.5 + 127.5
                if not os.path.exists(GENERATED_IMAGE_PATH):
                    os.mkdir(GENERATED_IMAGE_PATH)
                Image.fromarray(image.astype(np.uint8)).save(GENERATED_IMAGE_PATH + "%04d_%04d.png" % (epoch, index))

            # discriminator更新
            # 訓練データと生成画像をならべる

            X = np.concatenate((image_batch, generated_images))
            # 訓練データのラベルが1、生成画像のラベルが0になるように学習する
            y = [1] * BATCH_SIZE + [0] * BATCH_SIZE
            d_loss = discriminator.train_on_batch(X, y)

            # generator更新
            # generatorは騙したいのでノイズから生成した画像をDにいれたときに
            # Dの出力が1に近くなる（訓練画像と判定される確率が高い）ように学習する
            noise = np.array([np.random.uniform(-1, 1, 100) for _ in range(BATCH_SIZE)])
            g_loss = dcgan.train_on_batch(noise, [1] * BATCH_SIZE)

            print("epoch: %d, batch: %d, g_loss: %f, d_loss: %f" % (epoch, index, g_loss, d_loss))

    generator.save_weights('generator.h5')
    discriminator.save_weights('discriminator.h5')


if __name__ == '__main__':
    train()
