Kerasメモ
2016/10/18

# Kerasのインストール

http://machinelearningmastery.com/introduction-python-deep-learning-library-keras/

- backendにtheanoかtensorflowのインストールが必要
- どちらもWindowsでインストールするのは一苦労
- 参考までに自分の環境（Python3 Anaconda）
- GPUを使いたいときはAWSとかサクラか？
- VirtualBoxにUbuntuを入れて使っている
- keras.jsonの設定（バックエンドとimageの次元切り替え）

# Kerasによる2クラスロジスティック回帰

# Kerasで2値分類（Pima Indians Diabetes Data Set）

多層ニューラルネットワーク
http://machinelearningmastery.com/tutorial-first-neural-network-python-keras/

# Kerasで多値分類（Iris）

https://github.com/fastforwardlabs/keras-hello-world/blob/master/kerashelloworld.ipynb
http://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/

# Kerasで回帰問題

ロジスティック回帰は回帰とつくけど分類のアルゴリズム
出力が実数になる回帰を取り上げる

# MNIST

多層ニューラルネット
精度を比較

early_stopping
history plot

固定の100epochだとtest lossが上がっている（過学習！）
early_stoppingを使うと上がる前に打ち切れる

# CNN (MNIST)
# CNN (CIFAR-10)

# 学習済みモデルの保存

http://machinelearningmastery.com/save-load-keras-deep-learning-models/

# TensorFlowのラッパーライブラリ比較

- Keras
- TFLearn
- TF-slim

KerasはバックエンドにTensorFlowを使うと速度面でボトルネックあるようだ（速度比較）
Theanoと両対応が重荷になってきている？
しばらくは実務ではなくアルゴリズムの勉強目的だから実装例が多く、理解しやすいKerasで進める？
TFLearnの実装例はどの程度あるか？
https://www.reddit.com/r/MachineLearning/comments/50eokb/which_one_should_i_choose_keras_tensorlayer/
