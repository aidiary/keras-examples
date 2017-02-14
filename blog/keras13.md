畳み込みニューラルネットの可視化
2017/2/13

# はじめに

Deep Learningの学習結果（重み）はブラックボックスで隠れ層のユニットが一体何をやっているのかよくわからないと言われてきたが、ニューラルネットの可視化技術を使うことで徐々に光が当たってきている。

畳み込みニューラルネットで学習したフィルタの可視化というと以前やったようにフィルタの重みを直接可視化する方法がある。しかし、フィルタのサイズは基本的に数ピクセル（MNISTの例では5x5ピクセル）の小さな画像なのでこれを直接画像化しても何が学習されたか把握するのは難しい。たとえば、MNISTを学習した畳み込みニューラルネットのフィルタを可視化した例では各フィルタがどの方向に反応しやすいなどがわかる程度だ。

【画像：MNISTのフィルタの可視化結果】

各フィルタが何を学習したかを可視化する方法として**フィルタの出力を最大にする入力画像を生成する手法**が提案された。この生成画像は畳み込みニューラルネットへの入力画像と同じサイズなのでフィルタが最終的に何を学習したのかより把握しやすいというメリットがある。気持ち悪い画像を生成することで有名な[Deep Dream]()も基本的に今回紹介する可視化技術に基づいている。

今回は畳み込みニューラルネットとしてVGG16を対象にした（CIFAR-10を学習したCNNや最新のResNetもいずれやってみたい）。本家のKeras Blogにも[畳み込みニューラルネットワークのフィルタの可視化の記事](https://blog.keras.io/how-convolutional-neural-networks-see-the-world.html)があるが、この記事は少し古く`keras.applications.vgg16`モジュールが導入される前に書かれている。そこでこの記事では最新のKerasのvgg16モジュールを使うとともによりきれいな画像が生成されるように別の論文を参照していくつか工夫を加えてまとめることにした。

【画像】

# ナイーブな方法

個本的なアプローチは、畳み込みニューラルネットの指定したフィルタの出力を最大にする入力画像を**勾配法**を用いて更新する。数式で書くと

[tex: x \leftarrow x + \alpha \frac{\partial a_i (x)}{\partial x}]

となる。ここで、$x$は入力画像、$\alpha$は学習率、$a_i (x)$は画像$x$を入力したときのi番目のフィルタの出力（activation）だ。ここで、偏微分がいわゆる勾配を表し、入力画像$x$をちょっと変えたときにフィルタの出力がどれくらい変化するかを表している。入力画像$x$をこの勾配方向に徐々に移動する**勾配上昇法（gradient ascent）**を用いて**層の出力**を**最大化**する**入力画像を**求めている。

この式はニューラルネットの重みの更新式と比較するとわかりやすい。ニューラルネットの重みの更新式は

[tex: w \leftarrow w - \alpha \frac{\partial L(w)}{\partial w}]

ここで、$w$は重み、$\alpha$は学習率、$L(w)$は損失。ここで、偏微分は勾配を表し、重み$w$をちょっと変えたときに損失がどれくらい変化するかを表している。重み$w$をこの勾配の負の方向に徐々に移動する**勾配降下法（gradient descent）**を用いて**損失**を**最小化**する**重み**を求めている。先ほどは出力を最大化したいので勾配の正の方向に移動（ascent）したが、ニューラルネットの重み更新は損失を最小化したいがために勾配の負の方向に移動（descent）しているところが違うが両者の基本原理はよく似ている。

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

【図】勾配を伝搬する図

- この画像を入力したときに指定した層の指定したフィルタがもっとも活性化することを意味する
- つまり、このニューロンが見たがっている画像がこれということになる

実際は指定したクラスとは似ても似つかないむちゃくちゃな画像が生成されることが多い。しかし、そのむちゃくちゃな画像を入力しても畳み込みニューラルネットは99%以上の確率でそのクラスと認識してしまうのだ！詳細は、「畳み込みニューラルネットは簡単にだませる」という論文を参照。



このナイーブな方法ではあまりっきりとした画像が生成できなかったが、いくつかの改良を加えるとよりきれいで鮮やかな画像が生成できることが知られている。自然な画像らしさを表す事前知識（natural image priors）を正則化項（極端なピクセル値を取らないなど）として導入する。

# 工夫1: VGG16の出力を線形に

クラスの出力が0から1なのですぐに飽和して勾配が伝搬しにくい。一方、線形にすると出力が大きくなるので勾配が伝搬しやすくなる。

VGG16のsoftmaxの活性化関数をかける前の出力（つまり線形出力）だけ取りたかったのだがそのような関数は用意されていない。
もしsoftmax関数がLayerとして実装されていればその前の層の出力が取れたが、VGG16ではLayerではなく引数として実装されていた。
仕方ないのでここを参考に一部修正したVGG16クラスを用意した。

# 工夫2: 自然画像らしさを表す正則化項の導入

## Lp norm正則化

Lpノルムの定義
https://ja.wikipedia.org/wiki/Lp%E7%A9%BA%E9%96%93

画像のノルムを求めているのでピクセル値が極端に小さな値や大きな値になったときにブレーキをかけるような正則化
ピクセルのabsを取っているのに注目。小さな値もダメ！

```python
def normalize(img, value):
    return value / np.prod(K.int_shape(img)[1:])

# Lpノルム正則化項
# 今回の設定ではactivationは大きい方がよいため正則化のペナルティ項は引く
p = 6.0
lpnorm_weight = 10.0
if np.isinf(p):
    lp = K.max(input_tensor)
else:
    lp = K.pow(K.sum(K.pow(K.abs(input_tensor), p)), 1.0 / p)
activation -= lpnorm_weight * normalize(input_tensor, lp)
```

## Total Variation正則化

はじめて聞いたけど画像処理の分野では割と有名なようだ

Total Variationの定義
http://convexbrain.osdn.jp/cgi-bin/wifky.pl?p=Total+Variation
Total Variationが大きいほど隣通しのピクセルの差分が大きな画像になる
Total Variationが大きいほどブレーキがかかるようになっているため画像をぼやけさせる正則化と考えられる
aは画像の行方向の差分の大きさ
bは画像の列方向の差分の大きさ

```python
# Total Variationによる正則化
beta = 2.0
tv_weight = 10.0
a = K.square(input_tensor[:, 1:, :-1, :] - input_tensor[:, :-1, :-1, :])
b = K.square(input_tensor[:, :-1, 1:, :] - input_tensor[:, :-1, :-1, :])
tv = K.sum(K.pow(a + b, beta / 2.0))
activation -= tv_weight * normalize(input_tensor, tv)
```

【画像】softmax
【画像】linear
【画像】Lpノルム正則化
【画像】TV正則化

# 結果

- block1_conv1だけ正規化項を入れるとnanになってしまうので外した
- 他の層は入れたほうが鮮やかな画像になりやすい、入れないと少しくすんだ感じになる
- block5_conv3はなかなか難しい多くのニューロンがぼやっとした画像しか生成しない

【画像】ConvNetの図と各フィルタの画像を配置


畳み込みニューラルネットの可視化の原理がわかったので次はいよいよDeep Dreamを実装しよう！

# 参考


- [How convolutional neural networks see the world](https://blog.keras.io/how-convolutional-neural-networks-see-the-world.html)
- [Visualizing CNN filters with keras](https://jacobgil.github.io/deeplearning/filter-visualizations)
- [Understanding Neural Networks Through Deep Visualization](http://yosinski.com/deepvis)
- [Visualizing Deep Neural Networks Classes and Features](http://ankivil.com/visualizing-deep-neural-networks-classes-and-features/)
- [Meaning of Weight Gradient in CNN](http://stackoverflow.com/questions/38135950/meaning-of-weight-gradient-in-cnn)
- D. Erhan et al. [Visualizing Higher-Layer Features of a Deep Network](http://igva2012.wikispaces.asu.edu/file/view/Erhan+2009+Visualizing+higher+layer+features+of+a+deep+network.pdf)
- K. Simonyan et al. [Deep Inside Convolutional Networks: Visualising Image Classification Models and Saliency Maps](https://arxiv.org/abs/1312.6034)
- J. Yosinski et al. [Understanding Neural Networks Through Deep Visualization](http://yosinski.com/media/papers/Yosinski__2015__ICML_DL__Understanding_Neural_Networks_Through_Deep_Visualization__.pdf)
- A. Nguyen et al. [Deep Neural Networks are Easily Fooled: High Confidence Predictions for Unrecognizable Images](https://arxiv.org/abs/1412.1897)
