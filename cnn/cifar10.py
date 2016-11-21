import os
import numpy as np
from scipy.misc import toimage
import matplotlib.pyplot as plt

from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import SGD
from keras.utils import np_utils
from keras.utils.visualize_util import plot


def plot_cifar10(X, y):
    plt.figure()

    # 画像を描画
    nclasses = 10
    pos = 1
    for targetClass in range(nclasses):
        targetIdx = []
        # クラスclassIDの画像のインデックスリストを取得
        for i in range(len(y)):
            if y[i][0] == targetClass:
                targetIdx.append(i)

        # 各クラスからランダムに選んだ最初の10個の画像を描画
        np.random.shuffle(targetIdx)
        for idx in targetIdx[:10]:
            img = toimage(X[idx])
            plt.subplot(10, 10, pos)
            plt.imshow(img)
            plt.axis('off')
            pos += 1

    plt.savefig('result_cifar10/plot.png')


def plot_history(history, outdir):
    # 精度の履歴をプロット
    plt.figure()
    plt.plot(history.history['acc'], marker='.')
    plt.plot(history.history['val_acc'], marker='.')
    plt.title('model accuracy')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.grid()
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig(os.path.join(outdir, 'acc.png'))

    # 損失の履歴をプロット
    plt.figure()
    plt.plot(history.history['loss'], marker='.')
    plt.plot(history.history['val_loss'], marker='.')
    plt.title('model loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.grid()
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig(os.path.join(outdir, 'loss.png'))


if __name__ == '__main__':
    batch_size = 128
    nb_classes = 10
    nb_epoch = 100
    data_augmentation = False

    # 入力画像の次元
    img_rows, img_cols = 32, 32

    # チャネル数（RGBなので3）
    img_channels = 3

    # CIFAR-10データをロード
    # (nb_samples, nb_rows, nb_cols, nb_channel) = tf
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()

    # ランダムに画像をプロット
    plot_cifar10(X_train, y_train)

    # 画素値を0-1に変換
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255.0
    X_test /= 255.0

    # クラスラベル（0-9）をone-hotエンコーディング形式に変換
    Y_train = np_utils.to_categorical(y_train, nb_classes)
    Y_test = np_utils.to_categorical(y_test, nb_classes)

    # CNNを構築
    model = Sequential()

    model.add(Convolution2D(32, 3, 3, border_mode='same', input_shape=X_train.shape[1:]))
    model.add(Activation('relu'))
    model.add(Convolution2D(32, 3, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Convolution2D(64, 3, 3, border_mode='same'))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, 3, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    # モデルのサマリを表示
    model.summary()
    plot(model, show_shapes=True, to_file='result_cifar10/model.png')

    if not data_augmentation:
        print('Not using data augmentation')
        history = model.fit(X_train, Y_train,
                    batch_size=batch_size,
                    nb_epoch=nb_epoch,
                    verbose=1,
                    validation_split=0.1)

    # TODO: 学習したモデルの保存

    # 学習履歴をプロット
    plot_history(history, 'result_cifar10')

    # モデルの評価
    loss, acc = model.evaluate(X_test, Y_test, verbose=0)
    print('Test loss:', loss)
    print('Test acc:', acc)
