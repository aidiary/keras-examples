import numpy as np
import matplotlib.pyplot as plt
from scipy.misc import toimage
from keras.datasets import cifar10


def draw(X):
    """Xは4次元テンソルの画像集合、最初の16枚の画像を描画する"""
    assert X.shape[0] >= 16

    plt.figure()
    pos = 1
    for i in range(16):
        plt.subplot(4, 4, pos)
        img = toimage(X[i])
        plt.imshow(img)
        plt.axis('off')
        pos += 1
    plt.show()


# CIFAR10画像をロード（Xは4次元テンソル）
# 訓練画像集合5万枚のみ対象とする、他は捨てる
(X, _), (_, _) = cifar10.load_data()
print(X.shape)
# draw(X)

# 画像の枚数
N = X.shape[0]

# 画素値を0-1に正規化
X = X / 255.0
# 0-1に正規化しても同じように描画される
# draw(X)

# 全画像の画素値の平均を引く
X -= np.mean(X, axis=0)
# 値が負になっても同じように描画される
# draw(X)

# 画像とチャネルを1次元の行列に変換
# 1枚の画像が3072次元ベクトルとみなす
# (50000, 3072 = 32*32*3)
X = X.reshape(50000, -1)
print(X.shape)

# 共分散行列の固有値分解
# C => (3072, 3072)
# U => (3072, 3072)
# lam => (3072, )
# V => (3072, 3072)
C = np.dot(X.T, X) / N
U, S, V = np.linalg.svd(C)

# print("C:", C.shape)
# print("U:", U.shape)
# print("lam:", lam.shape)
# print("V:", V.shape)

# ZCA白色化
eps = 10e-7
sqS = np.sqrt(S + eps)
temp = np.dot(U, np.diag(1.0 / sqS))
# Uzca => (3072, 3072)
Uzca = np.dot(temp, U.T)

# ZはZCA白色化適用後の画像
# Z => (50000, 3072)
Z = np.dot(X, Uzca.T)

# 白色化した画像を描画
Z = Z.reshape(50000, 32, 32, 3)
draw(Z)
