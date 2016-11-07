Kerasによる多クラス分類（Iris）

2016/11/1

今回は、機械学習でよく使われるIrisデータセットを多層パーセプトロンで分類してみた。Irisデータセットのクラスラベルは3つあるので前回までと違って多クラス分類になる。短いプログラムなので全部載せてポイントだけまとめておこう。

```python
import numpy as np

from sklearn import datasets
from sklearn.model_selection import train_test_split

from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.utils import np_utils
from sklearn import preprocessing

def build_multilayer_perceptron():
    """多層パーセプトロンモデルを構築"""
    model = Sequential()
    model.add(Dense(16, input_shape=(4, )))
    model.add(Activation('relu'))
    model.add(Dense(3))
    model.add(Activation('softmax'))
    return model

if __name__ == "__main__":
    # Irisデータをロード
    iris = datasets.load_iris()
    X = iris.data
    Y = iris.target

    # データの標準化
    X = preprocessing.scale(X)

    # ラベルをone-hot-encoding形式に変換
    # 0 => [1, 0, 0]
    # 1 => [0, 1, 0]
    # 2 => [0, 0, 1]
    Y = np_utils.to_categorical(Y)

    # 訓練データとテストデータに分割
    train_X, test_X, train_Y, test_Y = train_test_split(X, Y, train_size=0.8)
    print(train_X.shape, test_X.shape, train_Y.shape, test_Y.shape)

    # モデル構築
    model = build_multilayer_perceptron()
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # モデル訓練
    model.fit(train_X, train_Y, nb_epoch=50, batch_size=1, verbose=1)

    # モデル評価
    loss, accuracy = model.evaluate(test_X, test_Y, verbose=0)
    print("Accuracy = {:.2f}".format(accuracy))
```

# ポイント

- irisデータは `sklearn.datasets.load_iris()` でダウンロードできる。よく使う標準データセットはメソッドが用意されていて便利

- irisのラベルは0, 1, 2のように数値ラベルになっている。これをニューラルネットで扱いやすい**one-hotエンコーディング**型式に変換する。上のコメントにも書いたようにone-hotエンコーディングは、特定のユニットのみ1でそれ以外は0のようなフォーマットのこと。この変換は、`keras.utils.np_utils` の `to_categorical()` に実装されている

- 訓練データとテストデータは、`sklearn.model_selection` の`train_test_split()` を使う。訓練データが8割、テストデータが2割になるように分割した。少し古いscikit-learnだと ` sklearn.cross_validation` に実装されているけどこのモジュールは**deprecated**

- モデル構築は `build_multilayer_perceptron()` という独自関数を用意した。モデルが複雑になると分けたほうがわかりやすい。入力層が4ユニット、隠れ層が16ユニット、出力層が3ユニットの多層パーセプトロンを構築した

- 多クラス分類の場合は、損失関数に `categorical_crossentropy` を指定する

必要な前処理メソッドはたいていscikit-learnに実装されているので自分で実装する前に探してみたほうがよさそう。

5回くらい実行してテストデータの精度を求めると97%、93%、100%、93%、97%となった。どのサンプルがテストデータに選ばれるかによって精度が変わるようだ。分類境界あたりのサンプルがテストデータとして選ばれると予測が難しくなり精度が下がりそうなのはわかる。

こういう場合は、何度か実行して平均を求めるのがセオリーかな。でもDeep Learningだと学習に何日もかかるケースがありそうだし、そんなに何回もデータセットの分割を変えて実行はできないなあ。どうするのが一般的なんだろうか？

# 参考

- ["Hello world" in keras](https://github.com/fastforwardlabs/keras-hello-world/blob/master/kerashelloworld.ipynb)
