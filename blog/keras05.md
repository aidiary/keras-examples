Kerasによる多クラス分類（Iris）

2016/11/1

今回は、機械学習でよく使われるIrisデータセットを使ってロジスティック回帰と多層パーセプトロンの精度を比較してみたい。Irisデータセットのクラスラベルは3つあるので前回までと違って多クラス分類になる。

# 1. Irisデータのロード

```python
# Irisデータをロード
iris = datasets.load_iris()
X = iris.data
Y = iris.target
```

# 2. one-hot-encoding

```python
# ラベルをone-hot-encoding形式に変換
# 0 => [1, 0, 0]
# 1 => [0, 1, 0]
# 2 => [0, 0, 1]
Y = np_utils.to_categorical(Y)
```

# 3. 訓練データとテストデータに分割

```python
# 訓練データとテストデータに分割
train_X, test_X, train_Y, test_Y = train_test_split(X, Y, train_size=0.8)
print(train_X.shape, test_X.shape, train_Y.shape, test_Y.shape)
```

# 4. 多層パーセプトロン

【図】

```python
# モデル構築
model = build_multilayer_perceptron()
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
```

```python
def build_multilayer_perceptron():
    """多層パーセプトロンモデルを構築"""
    model = Sequential()
    model.add(Dense(16, input_shape=(4, )))
    model.add(Activation('relu'))
    model.add(Dense(3))
    model.add(Activation('softmax'))
    return model
```

どのサンプルがテストに振り分けられるかで精度がけっこう変わってしまう。精度100%のときもあるけど、実際は精度97%くらいかな。
