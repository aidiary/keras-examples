import os
import numpy as np
import matplotlib.pyplot as plt

from sklearn import preprocessing

from keras.models import Sequential
from keras.layers.core import Dense, Activation

"""2クラスロジスティック回帰"""

def plot_data(X, t):
    # ラベル1を正例、ラベル0を負例とする
    positive = [i for i in range(len(t)) if t[i] == 1]
    negative = [i for i in range(len(t)) if t[i] == 0]

    plt.scatter(X[positive, 0], X[positive, 1], c='red', marker='o', label='positive')
    plt.scatter(X[negative, 0], X[negative, 1], c='blue', marker='o', label='negative')

if __name__ == '__main__':
    # 結果を再現できるように乱数の種を固定
    seed = 1
    np.random.seed(seed)

    # 訓練データをロード
    data = np.genfromtxt(os.path.join('..', 'data', 'ex2data1.txt'), delimiter=',')
    X = data[:, (0, 1)]
    t = data[:, 2]

    # データの各次元が平均0、分散1になるように正規化
    X = preprocessing.scale(X)

    # 訓練データをプロット
    plt.figure(1)
    plot_data(X, t)

    # ロジスティック回帰モデルを構築
    model = Sequential()
    model.add(Dense(1, input_shape=(2, )))
    model.add(Activation('sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # 訓練
    model.fit(X, t, nb_epoch=1000, batch_size=5, verbose=1)

    # 学習した重みを取得
    weights = model.layers[0].get_weights()
    w1 = weights[0][0, 0]
    w2 = weights[0][1, 0]
    b = weights[1][0]

    # 決定境界を描画
    plt.figure(1)
    xmin, xmax = min(X[:, 0]), max(X[:, 0])
    ymin, ymax = min(X[:, 1]), max(X[:, 1])
    xs = np.linspace(xmin, xmax, 100)
    ys = [- (w1 / w2) * x - (b / w2) for x in xs]
    plt.plot(xs, ys, 'b-', label='decision boundary')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)
    plt.legend()
    plt.show()
