畳み込みニューラルネットの可視化
2017/2/13

# はじめに

- 畳み込みニューラルネットで学習するフィルタはサイズが小さいため直接可視化してもあまり面白くない
- MNISTのようにどの方向に反応しやすいなどがわかる程度
- フィルタの出力をもっとも活性化する入力画像を生成する手法が提案された！
- 各フィルタがどのような入力画像にもっとも反応するかがわかる
- Deep Dreamの基本原理はこれ

# ナイーブな方法

- 画像に勾配を足し込む（gradient ascent）
- CNNの重み更新式と比べるとよくわかる


# 正則化の導入

# 隠れ層の可視化

# 出力層の可視化

# 参考

- [How convolutional neural networks see the world](https://blog.keras.io/how-convolutional-neural-networks-see-the-world.html)
- [Visualizing CNN filters with keras](https://jacobgil.github.io/deeplearning/filter-visualizations)
- [Understanding Neural Networks Through Deep Visualization](http://yosinski.com/deepvis)
- [Visualizing Deep Neural Networks Classes and Features](http://ankivil.com/visualizing-deep-neural-networks-classes-and-features/)
- [Meaning of Weight Gradient in CNN](http://stackoverflow.com/questions/38135950/meaning-of-weight-gradient-in-cnn)
- D. Erhan et al. [Visualizing Higher-Layer Features of a Deep Network](http://igva2012.wikispaces.asu.edu/file/view/Erhan+2009+Visualizing+higher+layer+features+of+a+deep+network.pdf)
- K. Simonyan et al. [Deep Inside Convolutional Networks: Visualising Image Classification Models and Saliency Maps](https://arxiv.org/abs/1312.6034)
