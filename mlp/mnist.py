import numpy as np
import matplotlib.pyplot as plt

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from keras.utils import np_utils
from keras.utils.visualize_util import plot

# MNISTの数字分類
# 参考
# https://github.com/fchollet/keras/blob/master/examples/mnist_mlp.py

def build_multilayer_perceptron():
    model = Sequential()

    model.add(Dense(512, input_shape=(784,)))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Dense(10))
    model.add(Activation('softmax'))

    return model

def plot_history(history):
    # print(history.history.keys())

    # 精度の履歴をプロット
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.legend(['acc', 'val_acc'], loc='lower right')
    plt.show()

    # 損失の履歴をプロット
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(['loss', 'val_loss'], loc='lower right')
    plt.show()

if __name__ == "__main__":
    batch_size = 128
    nb_classes = 10
    nb_epoch = 100

    # MNISTデータのロード
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    # 画像を1次元配列化
    X_train = X_train.reshape(60000, 784)
    X_test = X_test.reshape(10000, 784)

    # 画素を0.0-1.0の範囲に変換
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255
    X_test /= 255

    print(X_train.shape[0], 'train samples')
    print(X_test.shape[0], 'test samples')

    # one-hot-encoding
    Y_train = np_utils.to_categorical(y_train, nb_classes)
    Y_test = np_utils.to_categorical(y_test, nb_classes)

    # 多層ニューラルネットワークモデルを構築
    model = build_multilayer_perceptron()

    # モデルのサマリを表示
    model.summary()
    plot(model, show_shapes=True, show_layer_names=True, to_file='model.png')

    # モデルをコンパイル
    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(),
                  metrics=['accuracy'])

    # Early-stopping
    early_stopping = EarlyStopping(patience=0, verbose=1)

    # モデルの訓練
    history = model.fit(X_train, Y_train,
                        batch_size=batch_size,
                        nb_epoch=nb_epoch,
                        verbose=1,
                        validation_split=0.1,
                        callbacks=[early_stopping])

    # 学習履歴をプロット
    plot_history(history)

    # モデルの評価
    loss, acc = model.evaluate(X_test, Y_test, verbose=0)

    print('Test loss:', loss)
    print('Test acc:', acc)
