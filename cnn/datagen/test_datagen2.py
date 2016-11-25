import os
import shutil
import numpy as np
from scipy.misc import toimage
import matplotlib.pyplot as plt
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator


def draw(X, filename):
    plt.figure()
    pos = 1
    for i in range(X.shape[0]):
        plt.subplot(4, 4, pos)
        img = toimage(X[i])
        plt.imshow(img)
        plt.axis('off')
        pos += 1
    plt.savefig(filename)


if __name__ == '__main__':
    img_rows, img_cols, img_channels = 32, 32, 3
    batch_size = 16
    nb_classes = 10

    # CIFAR-10データをロード
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()

    # 画素値を0-1に変換
    X_train = X_train.astype('float32')
    X_train /= 255.0
    X_train = X_train[0:batch_size]
    y_train = y_train[0:batch_size]

    draw(X_train, 'datagen_before.png')

    # データ拡張
    datagen = ImageDataGenerator(
        rotation_range=90,
        zca_whitening=True
    )

    datagen.fit(X_train)
    g = datagen.flow(X_train, y_train, batch_size, shuffle=False)
    batch = g.next()
    print(batch[0].shape)
    print(batch[1].shape)

    draw(batch[0], 'datagen_after.png')
