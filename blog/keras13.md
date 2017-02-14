畳み込みニューラルネットの可視化
2017/2/13

# はじめに

畳み込みニューラルネットで学習したフィルタの可視化というと以前やったようにフィルタの重みを直接可視化する方法がある。しかし、フィルタのサイズは基本的に数ピクセル（MNISTの例では5x5ピクセル）の小さな画像なのでこれを直接画像化しても何が学習されたか把握するのは難しい。たとえば、MNISTを学習した畳み込みニューラルネットのフィルタを可視化した例では各フィルタがどの方向に反応しやすいなどがわかる程度だ。

【画像：MNISTのフィルタの可視化結果】

各フィルタが何を学習したかを可視化する方法として**フィルタの出力を最大にする入力画像を生成する手法**が提案された。この生成画像は畳み込みニューラルネットへの入力画像と同じサイズなのでフィルタが最終的に何を学習したのかより把握しやすいというメリットがある。気持ち悪い画像を生成することで有名なDeep Dreamも基本的に今回紹介する可視化技術に基づいている。

今回は畳み込みニューラルネットとしてVGG16を対象にした（CIFAR-10を学習したCNNでも面白そう）。本家のKeras Blogにも[畳み込みニューラルネットワークのフィルタの可視化の記事](https://blog.keras.io/how-convolutional-neural-networks-see-the-world.html)があるが、この記事は少し古く`keras.applications.vgg16`モジュールが導入される前に書かれている。そこでこの記事では最新のKerasのvgg16モジュールを使うとともによりきれいな画像が生成されるように別の論文を参照していくつか工夫を加えてまとめた。

【画像】

# ナイーブな方法

個本的なアプローチは、畳み込みニューラルネットの指定したフィルタの出力を最大にする入力画像を**勾配法**を用いて更新する。数式で書くと

[tex: x \leftarrow x + \alpha \frac{\partial a_i (x)}{\partial x}]

となる。ここで、$x$は入力画像、$\alpha$は学習率、$a_i (x)$は画像$x$を入力したときのi番目のフィルタの出力（activation）だ。ここで、偏微分がいわゆる勾配を表し、入力画像$x$をちょっと変えたときにフィルタの出力がどれくらい変化するかを表している。入力画像$x$をこの勾配方向に徐々に移動する勾配上昇法（gradient ascent）を用いて出力を**最大化**する**入力画像を**求めている。

この式はニューラルネットの重みの更新式と比較するとわかりやすい。ニューラルネットの重みの更新式は

[tex: w \leftarrow w - \alpha \frac{\partial L(w)}{\partial w}]

ここで、$w$は重み、$\alpha$は学習率、$L(w)$は損失。ここで、偏微分は勾配を表し、重み$w$をちょっと変えたときに損失がどれくらい変化するかを表している。重み$w$をこの勾配の負の方向に徐々に移動する勾配降下法（gradient descent）を用いて損失を**最小化**する**重み**を求めている。先ほどは出力を最大化したいので勾配の正の方向に移動したが、ニューラルネットの重み更新は損失を最小化したいがために勾配の負の方向に移動しているところが違うが構造はよく似ている。

ニューラルネットの重み更新では重みの初期値としてランダムな値を使った。フィルタの可視化でも同様に入力画像の初期値として下記のようなランダムな画像を使う（あとで紹介するDeep Dreamはこの画像の初期値として任意の画像を使っている）。

【開始時の画像】

Kerasで書くと下のようになる。

```python
# 指定した層の指定したフィルタの出力の平均
activation_weight = 1.0
if layer_name == 'predictions':
    # 出力層だけは2Dテンソル (num_samples, num_classes)
    activation = activation_weight * K.mean(layer.output[:, filter_index])
else:
    # 隠れ層は4Dテンソル (num_samples, row, col, channel)
    activation = activation_weight * K.mean(layer.output[:, :, :, filter_index])

# 層の出力の入力画像に対する勾配を求める
# 入力画像を微小量変化させたときの出力の変化量を意味する
# 層の出力を最大化したいためこの勾配を画像に足し込む
grads = K.gradients(activation, input_tensor)[0]

# 正規化トリック
# 画像に勾配を足し込んだときにちょうどよい値になる
grads /= (K.sqrt(K.mean(K.square(grads))) + K.epsilon())

# 画像を入力して層の出力と勾配を返す関数を定義
iterate = K.function([input_tensor], [activation, grads])

# ノイズを含んだ画像（4Dテンソル）から開始する
x = np.random.random((1, img_height, img_width, 3))
x = (x - 0.5) * 20 + 128

# 勾配法で層の出力（activation_value）を最大化するように入力画像を更新する
cache = None
for i in range(num_loops):
    activation_value, grads_value = iterate([x])
    # activation_valueを大きくしたいので画像に勾配を加える
    step, cache = rmsprop(grads_value, cache)
    x += step
    print(i, activation_value)

# 画像に戻す
img = deprocess_image(x[0])
```

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
