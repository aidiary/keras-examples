Kerasで二値分類（Pima Indians Diabetes Data Set）
2016/10/20

Kerasによる実験は、

1. データのロード
2. モデルの定義
3. モデルのコンパイル
4. モデルの学習
5. モデルの評価

という手続きを取る。
前回と違うデータでもう一回試してみたい。

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

最初の8列が患者の属性情報（意味はわからないｗ）で9列目が糖尿病を発症しない（0）または糖尿病を発症した（1）というラベルを表す。つまり、患者の属性情報から糖尿病を発症するかを予測するニューラルネットワークモデルを学習するのがここでの目的。

このデータは[CSV形式でダウンロードできる](http://archive.ics.uci.edu/ml/machine-learning-databases/pima-indians-diabetes/pima-indians-diabetes.data)。

データをロードするコードは、

```python
# load pima indians dataset
dataset = np.loadtxt(os.path.join("data", "pima-indians-diabetes.data"), delimiter=',')

# split into input (X) and output (Y) variables
X = dataset[:, 0:8]
Y = dataset[:, 8]
```

# 2. モデルの定義

# 3. モデルのコンパイル

# 4. モデルの学習

# 5. モデルの評価

# 参考

- [Develop Your First Neural Network in Python With Keras Step-By-Step](http://machinelearningmastery.com/tutorial-first-neural-network-python-keras/)
