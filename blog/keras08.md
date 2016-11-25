KerasでCIFAR-10の一般物体認識
2016/11/25

今回は、畳み込みニューラルネットを使って[CIFAR-10](http://aidiary.hatenablog.com/entry/20151014/1444827123)（2015/10/14）の一般物体認識をやってみた。以前、[Chainerでやった](http://aidiary.hatenablog.com/entry/20151108/1446952402)（2015/11/8）のをKerasで再実装してみた。

これも[Kerasの例題](https://github.com/fchollet/keras/blob/master/examples/cifar10_cnn.py)に含まれている。このスクリプトでは、データ拡張（Data Augmentation）も使っているがこれはまた別の回に取り上げよう。

# CIFAR-10

CIFAR-10は32x32ピクセル（ちっさ！）のカラー画像のデータセット。クラスラベルはairplane, automobile, bird, cat, deer, dog, frog, horse, ship, truckの10種類で訓練用データ50000枚、テスト用データ100000枚から成る。

【画像】

以前は、[CIFAR-10のホームページ](http://www.cs.toronto.edu/~kriz/cifar.html)から直接ダウンロードしたが、Kerasでは`keras.datasets.cifar10`モジュールを使えば勝手にダウンロードして使いやすい形で提供してくれる。


```python
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
```

`X_train`は (50000, 32, 32, 3) の4次元テンソルで与えられる（`image_dim_ordering`が`tf`のとき）。画像が50000枚、行数が32、列数が32、チャンネルが3（RGB）であることを意味する。配列には0-255の画素値が入っているため255で割って0-1に正規化する。`y_train`は (50000, 1) の配列で与えられる。クラスラベルは0-9の値が入っているのでNNで使いやすいようにone-hotエンコーディング形式に変換する。

# CNNの構築

少し層が深いCNNを構成してみた。

`INPUT -> ((CONV->RELU) * 2 -> POOL) * 2 -> FC`

畳み込み層（CONV）、ReLU活性化関数（RELU)を2回繰り返してプーリング層（POOL）を1セットとしてそれを2セット繰り返した後に全結合層（FC）を通して分類するという構成。

'''python
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
'''

# モデルの訓練

今回は、Early-stoppingは使わずに固定で100エポック回した。使うとすぐに収束扱いされてしまったため。使いどころがちょっと難しいかも。

```python
# 訓練
history = model.fit(X_train, Y_train,
                    batch_size=batch_size,
                    nb_epoch=nb_epoch,
                    verbose=1,
                    validation_split=0.1)

# 学習履歴をプロット
plot_history(history, result_dir)
```

損失と精度をプロットすると下のような感じ。

【結果】

CNNは学習にものすごい時間がかかる（GPUを使わないと特に）ので学習結果のモデルはファイルに保存するようにした。Kerasではモデルの形状（model.json）と学習した重み（model.h5）を別々に保存するようになっている。PythonなのにJSONを使うところがナウいと思った。h5というのはHDF5(Hierarchical Data Format)というバイナリフォーマットのようだ。ときどき見かけたけど使ったことがなかった。

```python
# 学習したモデルと重みと履歴の保存
model_json = model.to_json()
with open(os.path.join(result_dir, 'model.json'), 'w') as json_file:
    json_file.write(model_json)
model.save_weights(os.path.join(result_dir, 'model.h5'))
```

ちなみにファイルからモデルを読み込むときは

```python
model_file = os.path.join(result_dir, 'model.json')
weight_file = os.path.join(result_dir, 'model.h5')

with open(model_file, 'r') as fp:
    model = model_from_json(fp.read())
model.load_weights(weight_file)
model.summary()
```

とすればよい。

# モデルの評価

テストデータで評価すると **約80%** の分類精度が得られた。

```python
# モデルの評価
loss, acc = model.evaluate(X_test, Y_test, verbose=0)
print('Test loss:', loss)
print('Test acc:', acc)
```
