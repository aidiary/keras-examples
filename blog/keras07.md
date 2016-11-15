Kerasによる畳み込みニューラルネットワークの実装

2016/11/14

前回はMNISTの数字認識を多層パーセプトロンで解いたが、今回は畳み込みニューラルネットを使って解いてみた。このタスクも[Kerasの例題](https://github.com/fchollet/keras/blob/master/examples/mnist_cnn.py)に含まれている。ソースコードを見れば大体何をやっているかつかめそうだけどポイントを少しまとめておく。畳み込みニューラルネットワークの説明は、「ゼロから作るDeep Learning」の7章が非常にわかりやすいのでおすすめ。

# 4次元テンソルのチャネル位置

畳み込みニューラルネットでは、入力する画像の形状を保つために画像の集合を4次元テンソル（4次元配列）、すなわち画像の枚数、画像のチャネル数、画像の縦幅、画像の横幅で入力するのが一般的。Kerasでは、**4次元テンソルの各次元の位置が`image_dim_ordering`によって変わる** ので要注意。久々にKerasを動かしたら動かなくなっていてはまった。

`image_dim_ordering`は、`~/.keras/keras.json`で`th`（Theano）または`tf`（TensorFlow）のどちらかを指定できる。実際は、**バックエンドとは無関係に指定でき**、バックエンドにTensorFlowを使って`th`を指定してもよいし、バックエンドにTheanoを使って`tf`を指定してもよい。デフォルトでは`tf`のようだ。入力画像集合の各次元は

- `th`（Theano）では、**(サンプル数, チャンネル, 画像の行数, 画像の列数)**
- `tf`（TensorFlow）では、**（サンプル数, 画像の行数, 画像の列数, チャンネル）**

の並び順になる。例えば、`image_dim_ordering`が`tf`の場合、`X_train[sample][row][col][channel]`で画像の画素値にアクセスできる。両方に対応する場合は、下のようなコードを書く必要がある。keras.jsonの設定は`keras.backend`モジュールの`image_dim_ordering()`で取得できる。

```python
from keras import backend as K

# 画像集合を表す4次元テンソルに変形
# keras.jsonのimage_dim_orderingがthのときはチャネルが2次元目、tfのときはチャネルが4次元目にくる
if K.image_dim_ordering() == 'th':
    X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
    X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
    X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)
```

MNISTの入力数は白黒画像なのでチャネル数は1。

# 畳み込みニューラルネットの構築

畳み込み層が2つで、プーリング層が1つ、そのあとに多層パーセプトロンが続くシンプルな構成の畳み込みニューラルネットである。

【モデルの絵】

```python
def build_cnn(input_shape, nb_filters, filter_size, pool_size):
    model = Sequential()

    model.add(Convolution2D(nb_filters,
                            filter_size[0], filter_size[1],
                            border_mode='valid',
                            input_shape=input_shape))
    model.add(Activation('relu'))

    model.add(Convolution2D(nb_filters, filter_size[0], filter_size[1]))
    model.add(Activation('relu'))

    model.add(MaxPooling2D(pool_size=pool_size))
    model.add(Dropout(0.25))

    model.add(Flatten())

    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))

    return model
```

畳み込みニューラルネットでは、これまでの全結合層（`Dense`）以外に畳み込み層（`Convolution2D`）とプーリング層（`MaxPooling2D`）が出てくる。畳み込み層やプーリング層も「層」なので`model`に`add()`で追加できる。最後に多層パーセプトロンに入力するときはデータをフラット化する必要がある。4次元配列を1次元配列に変換するには`Flatten()`という層を追加するだけでOK。ユニット数などは自動的に計算してくれる。

モデルを図示してみると下のようになる。

【モデル図】

どうやらこの図の4次元テンソルは`image_dim_ordering`を`tf`にしていても`th`と同じ（サンプル数, チャネル数, 行数,　列数) になるようだ・・・ちゃんと `image_dim_ordering`の設定を見るようにしてほしいところ。

`Convolution2D`の`border_mode`を`valid`にすると出力画像は[入力画像より小さくなる](http://aidiary.hatenablog.com/entry/20150626/1435329581)（2015/6/26）。一方、`same`にすると自動的にパディングして出力画像が入力画像と同じサイズになるよう調整される。出力画像のサイズは計算式があるが、Kerasでは自動的に計算してくれている。

畳み込みニューラルネットのパラメータ数はフィルタのパラメータ数になる。例えば、最初の畳み込み層のパラメータ数は、[tex:32 \times 1 \times 3 \times 3 + 32 = 320] となる。32を足すのは各フィルタにあるスカラーのバイアス項。二つ目の畳み込み層のパラメータ数は、[tex:32 \times 32 \times 3 \times 3 + 32 = 9248]となる。

残りは前回の多層パーセプトロンとまったく同じなので省略。

実行すると7エポックほどで収束し、精度は99%近く出る。

# フィルタの可視化


# 参考

- [Handwritten Digit Recognition using Convolutional Neural Networks in Python with Keras](http://machinelearningmastery.com/handwritten-digit-recognition-using-convolutional-neural-networks-python-keras/)
- [How does border_mode work?](https://github.com/fchollet/keras/issues/1984)
- [How convolutional neural networks see the world](https://blog.keras.io/how-convolutional-neural-networks-see-the-world.html)
