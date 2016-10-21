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

http://machinelearningmastery.com/tutorial-first-neural-network-python-keras/


# TensorFlowのラッパーライブラリ比較

- Keras
- TFLearn
- TF-slim

KerasはバックエンドにTensorFlowを使うと速度面でボトルネックあるようだ（速度比較）
Theanoと両対応が重荷になってきている？
しばらくは実務ではなくアルゴリズムの勉強目的だから実装例が多く、理解しやすいKerasで進める？
TFLearnの実装例はどの程度あるか？
https://www.reddit.com/r/MachineLearning/comments/50eokb/which_one_should_i_choose_keras_tensorlayer/
