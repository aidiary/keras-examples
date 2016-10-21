Kerasで二値分類（Pima Indians Diabetes Data Set）

2016/10/20

Kerasのプログラミングは

1. データのロード
2. モデルの定義
3. モデルのコンパイル
4. モデルの学習
5. モデルの評価
6. 新データに対する予測

という手順が一般的。

# 1. データのロード

[UCI Machine Learning repository](http://archive.ics.uci.edu/ml/index.html)にある[Pima Indians Diabetes Data Set](http://archive.ics.uci.edu/ml/datasets/Pima+Indians+Diabetes)を使う。医療系のデータでPimaインディアンが糖尿病にかかったかどうかを表すデータのようだ。

```
Attribute Information:
1. Number of times pregnant
2. Plasma glucose concentration a 2 hours in an oral glucose tolerance test
3. Diastolic blood pressure (mm Hg)
4. Triceps skin fold thickness (mm)
5. 2-Hour serum insulin (mu U/ml)
6. Body mass index (weight in kg/(height in m)^2)
7. Diabetes pedigree function
8. Age (years)
9. Class variable (0 or 1)
```

最初の8列が患者の属性情報で9列目が糖尿病を発症しない（0）または糖尿病を発症した（1）というラベルを表す。つまり、患者の属性情報から糖尿病を発症するかを予測するニューラルネットワークモデルを学習するのがここでの目的。

このデータは[CSV形式でダウンロードできる](http://archive.ics.uci.edu/ml/machine-learning-databases/pima-indians-diabetes/pima-indians-diabetes.data)。データをロードするコードは、

```python
# load pima indians dataset
dataset = np.loadtxt(os.path.join("data", "pima-indians-diabetes.data"), delimiter=',')

# split into input (X) and output (Y) variables
X = dataset[:, 0:8]
Y = dataset[:, 8]
```

ここで、二次元データならグラフで可視化してみるところだけれど、8次元データはそのままでは可視化できない。分析するなら属性間の相関図や次元圧縮してみるのがセオリーか？今回はKerasの使い方の習得がメインなので飛ばそう。

# 2. モデルの定義

ここはニューラルネットの構造を組み立てるところ。
今回は、隠れ層が2つ、出力層が1つの多層ニューラルネットを構築する。

【図】

```python
# create model
model = Sequential()
model.add(Dense(12, input_dim=8, init='uniform', activation='relu'))
model.add(Dense(8, init='uniform', activation='relu'))
model.add(Dense(1, init='uniform', activation='sigmoid'))
```

- 入力層は独立した層ではなく、`input_dim`で指定する。
- 全結合層は`Dense`を使う。
- `init`で層の[重みの初期化方法](https://keras.io/ja/initializations/)を指定できる。
- `uniform`だと0～0.05の一様乱数。`normal`だと正規乱数。[Deep Learning Tutorialの初期値重み](http://aidiary.hatenablog.com/entry/20150618/1434628272)で使われていた`glorot_uniform`も実装されている。
- 層の活性化関数は、`activation`で指定。上の例では隠れ層にはReLU、出力層は`sigmoid`を指定。
- 出力層に`sigmoid`を使うと0.0から1.0の値が出力されるため、入力したデータのクラスが1である確率とみなせる。0.5を閾値として0.5未満ならクラス0、0.5以上ならクラス1として分類する。

# 3. モデルのコンパイル

損失関数、最適化関数、評価指標を指定してモデルをコンパイルする。

```python
# compile model
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
```

# 4. モデルの学習

```python
# fit the model
model.fit(X, Y, nb_epoch=150, batch_size=10)
```

- 訓練データ`X`とラベル`Y`を指定して学習する。
- エポックは固定で150回ループを回す。
- 学習はいわゆるミニバッチ学習でバッチサイズは10とした。データを10個ずつ取り出して重みを更新する。

# 5. モデルの評価

```python
# evaluate the model
scores = model.evaluate(X, Y)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))
```

- モデルの評価には`model.evaluate()`を使う。
- 戻り値は評価指標のリスト。デフォルトでは損失（loss）のみ。`compile()`の`metrics`に評価指標を追加すると、別の評価指標が追加される。今回は、精度（acc）を追加してある。model.metrics_namesで、評価尺度の名前リストが得られる。

```python
> print(model.metrics_names)
> print(scores)
['loss', 'acc']
[0.45379118372996646, 0.7890625]
```

- ここでは、訓練データを使って評価しているが、実際は訓練データとは独立した**テストデータを用意するべき**。
- ただ、モデルにバグがないか確認するために最初は訓練データで評価してみるのもありだと思う。訓練データでさえ精度がものすごく低かったらモデルに何か問題がある。
- 今回は、訓練データそのものの予測で精度79%

# 6. 新データに対する予測

```python
predictions = np.round(model.predict(X))
correct = Y[:, np.newaxis]
```

- 新しいデータを入力してモデルの出力を求めるには`model.predict()`を使う
- 今回は簡単のため訓練データ`X`をそのまま入力
- 出力層は`sigmoid`なので下のように0.0から1.0の値が出力される

```
[[ 0.82117373]
 [ 0.10473501]
 [ 0.90901828]
 [ 0.10512047]
 [ 0.78090101]
 ...
```

- これをクラスラベルにするために`round()`を使う。0.5以上なら1に切り上げ、未満なら0に切り下げしてくれる。

```
[[ 1.]
 [ 0.]
 [ 1.]
 [ 0.]
 [ 1.]
 ...
```

今回は実践的なデータセットを用いてKerasによる実験の流れを一通りまとめた。

# 参考

- [Develop Your First Neural Network in Python With Keras Step-By-Step](http://machinelearningmastery.com/tutorial-first-neural-network-python-keras/)
