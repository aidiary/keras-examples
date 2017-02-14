畳み込みニューラルネットの可視化
2017/2/13

# はじめに

畳み込みニューラルネットで学習したフィルタの可視化というと以前やったようにフィルタの重みを直接可視化する方法がある。しかし、フィルタのサイズは基本的に数ピクセル（MNISTの例では5x5ピクセル）の小さな画像なので何が学習されたか把握するのは少し難しい。たとえば、MNISTのようにどの方向に反応しやすいなどがわかる程度だ。

各フィルタが何を学習したかをよりよく可視化する方法として**フィルタの出力をもっとも活性化する入力画像を生成する手法**が発明された。この入力画像は畳み込みニューラルネットへの入力画像のサイズと同じなのでフィルタが最終的に何を学習したのかより把握しやすいというメリットがある。気持ち悪い画像を生成することで有名なDeep Dreamも基本的に今回紹介する可視化技術に基づいている。

畳み込みニューラルネットとしてCIFAR-10を学習したモデルでもよかったのだが、より面白そうなVGG16を対象にした。本家のKeras Blogにも[畳み込みニューラルネットワークのフィルタの可視化の記事](https://blog.keras.io/how-convolutional-neural-networks-see-the-world.html)があるが、この記事は少し古く`keras.applications.vgg16`モジュールが導入される前に書かれている。そこでこの記事では最新のKerasのvgg16モジュールを使うとともによりきれいな画像が生成されるように別の論文を参照していくつか工夫を加えてまとめた。

【画像】

# ナイーブな方法

- 開始時の画像【画像】
- 画像に勾配を足し込む（gradient ascent）
- CNNの重み更新式と比べるとよくわかる！
- この画像を入力したときに指定した層の指定したフィルタがもっとも活性化することを意味する


このナイーブな方法ではあまり下記のようにはっきりとした画像が生成できなかったがいくつかの工夫を加えるとよりきれいな画像が生成できることがわかった。ここでは、

- VGG16の出力層の活性化関数をsoftmaxからlinearに変更
- 最適化アルゴリズムをRMSpropに変更
- Lpノルム正則化の導入
- Total Variation正則化の導入

の4つの工夫を加えた。

# VGG16の修正


# 正則化の導入

- Lp norm正則化
- Total Variation正則化

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
