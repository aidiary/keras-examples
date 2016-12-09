import os
import sys
import numpy as np
from scipy.misc import toimage
import matplotlib.pyplot as plt

from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras.utils.visualize_util import plot
from keras.preprocessing.image import ImageDataGenerator


def plot_cifar10(X, y, result_dir):
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

    plt.savefig(os.path.join(result_dir, 'plot.png'))


def save_history(history, result_file):
    loss = history.history['loss']
    acc = history.history['acc']
    val_loss = history.history['val_loss']
    val_acc = history.history['val_acc']
    nb_epoch = len(acc)

    with open(result_file, "w") as fp:
        fp.write("epoch\tloss\tacc\tval_loss\tval_acc\n")
        for i in range(nb_epoch):
            fp.write("%d\t%f\t%f\t%f\t%f\n" % (i, loss[i], acc[i], val_loss[i], val_acc[i]))


if __name__ == '__main__':
    if len(sys.argv) != 4:
        print("usage: python cifar10.py [nb_epoch] [use_data_augmentation (True or False)] [result_dir]")
        exit(1)

    nb_epoch = int(sys.argv[1])
    data_augmentation = True if sys.argv[2] == "True" else False
    result_dir = sys.argv[3]
    if not os.path.exists(result_dir):
        os.mkdir(result_dir)

    print("nb_epoch:", nb_epoch)
    print("data_augmentation:", data_augmentation)
    print("result_dir:", result_dir)

    batch_size = 128
    nb_classes = 10

    # 入力画像の次元
    img_rows, img_cols = 32, 32

    # チャネル数（RGBなので3）
    img_channels = 3

    # CIFAR-10データをロード
    # (nb_samples, nb_rows, nb_cols, nb_channel) = tf
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()

    # ランダムに画像をプロット
    plot_cifar10(X_train, y_train, result_dir)

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
    plot(model, show_shapes=True, to_file=os.path.join(result_dir, 'model.png'))

    # 訓練
    if not data_augmentation:
        print('Not using data augmentation')
        history = model.fit(X_train, Y_train,
                            batch_size=batch_size,
                            nb_epoch=nb_epoch,
                            verbose=1,
                            validation_data=(X_test, Y_test),
                            shuffle=True)
    else:
        print('Using real-time data augmentation')

        # 訓練データを生成するジェネレータ
        train_datagen = ImageDataGenerator(zca_whitening=True, width_shift_range=0.1, height_shift_range=0.1)
        train_datagen.fit(X_train)
        train_generator = train_datagen.flow(X_train, Y_train, batch_size=batch_size)

        # テストデータを生成するジェネレータ
        # 画像のランダムシフトは必要ない？
        test_datagen = ImageDataGenerator(zca_whitening=True)
        test_datagen.fit(X_test)
        test_generator = test_datagen.flow(X_test, Y_test)

        # ジェネレータから生成される画像を使って学習
        # 本来は好ましくないがテストデータをバリデーションデータとして使う
        # validation_dataにジェネレータを使うときはnb_val_samplesを指定する必要あり
        # TODO: 毎エポックで生成するのは無駄か？
        history = model.fit_generator(train_generator,
                                      samples_per_epoch=X_train.shape[0],
                                      nb_epoch=nb_epoch,
                                      validation_data=test_generator,
                                      nb_val_samples=X_test.shape[0])

    # 学習したモデルと重みと履歴の保存
    model_json = model.to_json()
    with open(os.path.join(result_dir, 'model.json'), 'w') as json_file:
        json_file.write(model_json)
    model.save_weights(os.path.join(result_dir, 'model.h5'))
    save_history(history, os.path.join(result_dir, 'history.txt'))

    # モデルの評価
    # 学習は白色化した画像を使ったので評価でも白色化したデータで評価する
    if not data_augmentation:
        loss, acc = model.evaluate(X_test, Y_test, verbose=0)
    else:
        loss, acc = model.evaluate_generator(test_generator, val_samples=X_test.shape[0])

    print('Test loss:', loss)
    print('Test acc:', acc)
