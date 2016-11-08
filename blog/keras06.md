KerasでMNIST

2016/11/7

今回は、KerasでMNISTの数字認識をするプログラムを書いた。このタスクは、[Kerasの例題](https://github.com/fchollet/keras/blob/master/examples/mnist_mlp.py)にも含まれている。今まで使ってこなかったモデルの可視化、Early-stoppingによる収束判定、学習履歴のプロットなども取り上げてみた。

# MNISTデータのロードと前処理

MNISTをロードするモジュールはKerasで提供されているので使った。

```python
from keras.datasets import mnist
from keras.utils import np_utils

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

```

KerasでダウンロードしたMNISTのデフォルトの形状は `(60000, 28, 28)` なので `(60000, 784)` に `reshape`する。各サンプルが784次元ベクトルになるようにしている。画像データ（`X`）は0-255の画素値が入っているため0.0-1.0に正規化する。クラスラベル（`y`）は数字の0-9が入っているためone-hotエンコーディング型式に変換する。

`nb_classes = 10`を省略すると`y_test`に入っているラベルから自動的にクラス数を推定してくれるようだが、必ずしも0-9のラベルすべてが`y_test`に含まれるとは限らないため`nb_classes = 10`を指定したほうが安全のようだ。

# モデルの可視化

```python
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import Adam
from keras.utils.visualize_util import plot

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

# 多層ニューラルネットワークモデルを構築
model = build_multilayer_perceptron()

# モデルのサマリを表示
model.summary()
plot(model, show_shapes=True, show_layer_names=True, to_file='model.png')

# モデルをコンパイル
model.compile(loss='categorical_crossentropy',
              optimizer=Adam(),
              metrics=['accuracy'])
```

隠れ層が2つある多層パーセプトロンを構築した。活性化関数には `relu`。また、過学習を防止するテクニックである `Dropout` を用いた。Dropoutも層として追加する。

`model.summary()`を使うと以下のようなモデル形状のサマリが表示される。`model`に`add()`で追加した順になっていることがわかる。Output Shapeの`None`はサンプル数を表しているが省略できる。`dense_1` 層のパラメータ数（重み行列のサイズのこと）は 784*512+512=401920 となる。512を足すのは**バイアスも重みに含めている**ため。ユーザがバイアスの存在を気にする必要はないが、裏ではバイアスも考慮されていることがパラメータ数からわかる。同様に `dense_2` 層のパラメータ数は 512*512+512=262656 となる。同様に `dense_3` 層のパラメータ数は 512*10+10=5130 となる。

```text
____________________________________________________________________________________________________
Layer (type)                     Output Shape          Param #     Connected to
====================================================================================================
dense_1 (Dense)                  (None, 512)           401920      dense_input_1[0][0]
____________________________________________________________________________________________________
activation_1 (Activation)        (None, 512)           0           dense_1[0][0]
____________________________________________________________________________________________________
dropout_1 (Dropout)              (None, 512)           0           activation_1[0][0]
____________________________________________________________________________________________________
dense_2 (Dense)                  (None, 512)           262656      dropout_1[0][0]
____________________________________________________________________________________________________
activation_2 (Activation)        (None, 512)           0           dense_2[0][0]
____________________________________________________________________________________________________
dropout_2 (Dropout)              (None, 512)           0           activation_2[0][0]
____________________________________________________________________________________________________
dense_3 (Dense)                  (None, 10)            5130        dropout_2[0][0]
____________________________________________________________________________________________________
activation_3 (Activation)        (None, 10)            0           dense_3[0][0]
====================================================================================================
Total params: 669706
```

`keras.utils.visualize_util` の `plot()` を使うとモデルを画像として保存できる。今はまだ単純なモデルなので `summary()` と同じでありがたみがないがもっと複雑なモデルだと図の方がわかりやすそう。

【画像】

# Early-stoppingによる収束判定

これまでの実験では、適当に `nb_epoch` を決めて固定の回数だけ訓練ループを回していたが過学習の危険がある。たとえば、このMNISTタスクで100回回したときの `train_loss`（訓練データの損失）と `val_loss`（バリデーションセットの損失）をプロットすると下のようになる（次節の`history`でプロット）。

【図】

この図からわかるように `train_loss` はエポックが経つにつれてどんどん下がるが、逆に `val_loss` が上がっていくことがわかる。これは、訓練データセットに過剰にフィットしてしまうために未知のデータセットに対する予測性能が下がってしまう**過学習**を起こしていることを意味する。機械学習の目的は未知のデータセットに対する予測性能を上げることなので過学習はダメ！

普通は訓練ループを回すほど性能が上がりそうだけど、先に見たように**訓練ループを回せば回すほど性能が悪化する場合がある**。そのため、予測性能が下がる前にループを打ち切りたい。`val_loss` をプロットして目視でどこで打ち切るか判断することもできるが、それを自動で判断してくれるのがEarly-stopping（早期打ち切り）というアルゴリズム。

Kerasではコールバック関数として`EarlyStopping`が実装されているため`fit()`の`callbacks`オプションに設定する。**`EarlyStopping`を使うには必ずバリデーションデータセットを用意する必要がある**。`fit()`のオプションで`validation_data`を直接指定することもできるが、`validation_split`を指定することで**訓練データの一部をバリデーションデータセットとして使う**ことができる。

[Keras examples](https://github.com/fchollet/keras/blob/master/examples/mnist_mlp.py)もそうだが、テストデータセットをバリデーションデータセットとして使うのは本来ダメらしい。[バリデーションデータセットとテストデータセットは分けたほうがよい](http://stats.stackexchange.com/questions/19048/what-is-the-difference-between-test-set-and-validation-set)。

```python
# Early-stopping
early_stopping = EarlyStopping(patience=0, verbose=1)

# モデルの訓練
history = model.fit(X_train, Y_train,
                    batch_size=batch_size,
                    nb_epoch=nb_epoch,
                    verbose=1,
                    validation_split=0.1,
                    callbacks=[early_stopping])
```

`EarlyStopping`を導入するとわずか5エポックくらいで訓練が打ち切られる。確かにここら辺から`val_loss`が上がってくるのでよいのかもしれないが少し早すぎかもしれない。`EarlyStopping`には`patience`というパラメータを指定できる。デフォルトでは0だが、`patience`を上げていくと訓練を打ち切るのを様子見するようになる。通常はデフォルトでOKか？

verboseが機能していない？どこに表示されるのだろう？

# 学習履歴のプロット

`fit()`の戻り値である `history` に学習経過の履歴が格納されている。このオブジェクトを使えばいろいろな経過情報をプロットできる。デフォルトでは、`loss`（訓練データセットの損失）だけだが、`model`の`metrics`に`accuracy`を追加すると`acc`（精度）が、バリデーションデータセットを使うと`val_loss`（バリデーションデータセットの損失）や`val_acc`（バリデーションデータセットの精度）が自動的に追加される。ユーザ独自のメトリクスも定義できるようだ。

```python
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

# 学習履歴をプロット
plot_history(history)
```

今回は、Kerasの便利なツールをいろいろ使ってみた。

# 参考

- [Keras examples](https://github.com/fchollet/keras/blob/master/examples/mnist_mlp.py)
- [コールバックの使い方](https://keras.io/ja/callbacks/)
- [Display Deep Learning Model Training History in Keras](http://machinelearningmastery.com/display-deep-learning-model-training-history-in-keras/)
- [Dropout Regularization For Neural Networks](http://machinelearningmastery.com/dropout-regularization-deep-learning-models-keras/)
