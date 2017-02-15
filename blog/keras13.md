畳み込みニューラルネットの可視化
2017/2/13

# はじめに

Deep Learningの学習結果（重み）はブラックボックスで、隠れ層のユニット（特に深い層の！）が一体何を学習したのかがよくわからないと長年言われてきた。しかし、これから紹介するニューラルネットの可視化技術を使うことでそれらの理解が徐々に進んでいる。

畳み込みニューラルネットで学習したフィルタの可視化というと以前やったように学習した第1層のフィルタの重みを直接画像として可視化する方法がある。しかし、畳み込みフィルタのサイズは基本的に数ピクセル（MNISTの例では5x5ピクセル程度）のとても小さな画像なのでこれを直接可視化しても結局何が学習されたか把握するのはとても難しい。たとえば、MNISTを学習した畳み込みニューラルネットのフィルタを可視化した例では各フィルタがどの方向に反応しやすいかがわかる程度だ。

【画像：MNISTのフィルタの可視化結果】

各フィルタが何を学習したかを可視化する別のアプローチとして**各フィルタの出力を最大化するような入力画像を生成する手法**が提案された。この生成画像は**ニューラルネットへの入力画像と同じサイズ**なのでフィルタが最終的に何を学習したのかより把握しやすいというメリットがある。

今回は畳み込みニューラルネットの一種であるVGG16を対象に学習したフィルタを可視化してみたい。あとでMNISTやCIFAR-10を学習したCNNや最新のResNetでもやってみるつもり。

基本的に本家のKeras Blogの[畳み込みニューラルネットワークのフィルタの可視化の記事](https://blog.keras.io/how-convolutional-neural-networks-see-the-world.html)を参考にした。しかし、この記事で使われているKerasのバージョンは少し古く`keras.applications.vgg16`モジュールが導入される前に書かれている。また、正則化などの工夫が入っておらず生成される画像があまり美しくない。

そこで今回は最新のKerasの`vgg16`モジュールを使って書き換えるとともに、よりきれいな画像が生成されるようないくつか工夫を加えて結果をまとめることにした。

【画像】

# ナイーブな方法

基本的なアプローチは、畳み込みニューラルネットの指定したフィルタの出力を最大にする入力画像を**勾配法**を用いて更新する。数式で書くと

[tex: x \leftarrow x + \eta \frac{\partial a_i (x)}{\partial x}]

となる。ここで、$x$は入力画像、$\eta$は学習率、$a_i (x)$は画像$x$を入力したときのi番目のフィルタの出力（activation）だ。ここで、偏微分がいわゆる勾配を表し、入力画像$x$をちょっと変えたときにフィルタの出力がどれくらい変化するかを表している。入力画像$x$をこの勾配方向に徐々に移動する**勾配上昇法（gradient ascent）**を用いて**層の出力**を**最大化**する**入力画像を**求めている。

この式はニューラルネットの重みの更新式と比較するとわかりやすい。ニューラルネットの重みの更新式は

[tex: w \leftarrow w - \eta \frac{\partial L(w)}{\partial w}]

ここで、$w$は重み、$\eta$は学習率、$L(w)$は損失。ここで、偏微分は勾配を表し、重み$w$をちょっと変えたときに損失がどれくらい変化するかを表している。重み$w$をこの勾配の負の方向に徐々に移動する**勾配降下法（gradient descent）**を用いて**損失**を**最小化**する**重み**を求めている。

先ほどは出力を最大化したいので勾配の正の方向に移動（ascent）したが、ニューラルネットの重み更新は損失を最小化したいがために勾配の負の方向に移動（descent）しているところと更新する対象が異なるが両者の基本原理はよく似ていることがわかる。

ニューラルネットの重み更新では重みの初期値としてランダムな値を使った。フィルタの可視化でも同様に入力画像の初期値として下記のようなランダムな画像を使う（Deep Dreamはこの画像の初期値として任意の画像を使っている）。

【initial_image.png】

Kerasで書くと下のようになる。

```python
def visualize_filter(layer_name, filter_index, num_loops=200):
    """指定した層の指定したフィルタの出力を最大化する入力画像を勾配法で求める"""
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

    return img
```

ここで指定するフィルタは畳み込みニューラルネット内の任意の層（出力層含む）を指定できる。

```python
visualize_filter('predictions', 65)
```

フィルタの名前は`model.summary()`で表示されるKeras Layerの名前である。`predictions`は`VGG16`の出力層に付けられた名前。出力層の65番目のユニットは sea snake（ウミヘビ）クラスを意味する。1個ではつまらないので16個まとめて画像化すると下のようになる。

【vgg16_sigmoid.png】

まあこの可視化画像を見てもはっきり言ってよくわからない。でもこれらの画像を入力するとウミヘビである確率は99%という出力が出てくる。**人間にはまったくウミヘビと認識できない画像なのに、ニューラルネットは99%の確率でウミヘビと判定してしまう**。ここら辺は参考文献で挙げた「ニューラルネットは簡単にだませる」という論文を参照。

このナイーブな方法ではあまりっきりとした画像が生成できなかったが、いくつかの改良を加えるとよりはっきりとした鮮やかな画像が生成できることが知られている。そのために自然な画像らしさを表す事前知識（natural image priors）を正則化項として導入する。

# 工夫1: VGG16の出力を線形に

VGG16の出力層の出力は分類のために`softmax`活性化関数を適用しており、1000クラスの合計が1.0になるように正規化されている。この場合、あるクラスの出力を最大化するためにそのクラスの出力を上げる以外に、他のクラスの出力を下げるという方法もとれてしまう。このせいで最適化が惑わされてうまくいかないという指摘がある（Simonyan 2013）。そこで、最初の工夫として`softmax`を`linear`にしてみた。

VGG16のsoftmaxの活性化関数をかける前の出力（つまり線形出力）が取れれば簡単なのだがKerasにそのようなプロパティは用意されていなかった。もしVGG16の`softmax`活性化関数が

```python
model.add(Dense(1000))
model.add(Activation('softmax'))
```

のように独立した層として追加されていれば、一つ前の`softmax`をかける前の出力は簡単に取れたのだが、VGG16は下のように層ではなく引数として実装されていた。

```python
model.add(Dense(1000, activation='softmax'))
```

これだと`softmax`をかける前の出力は取り出せない・・・仕方ないのでKerasのVGG16をコピーして一部修正したVGG16クラスを用意した。

```python
if include_top:
    # Classification block
    x = Flatten(name='flatten')(x)
    x = Dense(4096, activation='relu', name='fc1')(x)
    x = Dense(4096, activation='relu', name='fc2')(x)
    # ここの活性化関数をsoftmaxからlinearに置き換え
    x = Dense(1000, activation='linear', name='predictions')(x) <= ここをlinearに
```

【画像】vgg16_linear.png

# 工夫2: Lp norm正則化

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

【画像】lpnorm.png

# 工夫3: Total Variation正則化

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

【画像】tv.png

# 結果

- block1_conv1だけ正規化項を入れるとnanになってしまうので外した
- 他の層は入れたほうが鮮やかな画像になりやすい、入れないと少しくすんだ感じになる
- block5_conv3はなかなか難しい多くのニューロンがぼやっとした画像しか生成しない

【画像】ConvNetの図と各フィルタの画像を配置


畳み込みニューラルネットの可視化の原理がわかったので次はいよいよ気持ち悪い画像を生成することで有名なDeep Dreamを実装しよう！Deep Dreamも基本的に今回紹介する可視化技術に基づいている。

# 参考


- [How convolutional neural networks see the world](https://blog.keras.io/how-convolutional-neural-networks-see-the-world.html)
- [Visualizing CNN filters with keras](https://jacobgil.github.io/deeplearning/filter-visualizations)
- [Understanding Neural Networks Through Deep Visualization](http://yosinski.com/deepvis)
- [Visualizing Deep Neural Networks Classes and Features](http://ankivil.com/visualizing-deep-neural-networks-classes-and-features/)
- [Meaning of Weight Gradient in CNN](http://stackoverflow.com/questions/38135950/meaning-of-weight-gradient-in-cnn)
- D. Erhan et al. [Visualizing Higher-Layer Features of a Deep Network](http://igva2012.wikispaces.asu.edu/file/view/Erhan+2009+Visualizing+higher+layer+features+of+a+deep+network.pdf) (2009)
- K. Simonyan et al. [Deep Inside Convolutional Networks: Visualising Image Classification Models and Saliency Maps](https://arxiv.org/abs/1312.6034) (2013)
- J. Yosinski et al. [Understanding Neural Networks Through Deep Visualization](http://yosinski.com/media/papers/Yosinski__2015__ICML_DL__Understanding_Neural_Networks_Through_Deep_Visualization__.pdf) (2015)
- A. Nguyen et al. [Deep Neural Networks are Easily Fooled: High Confidence Predictions for Unrecognizable Images](https://arxiv.org/abs/1412.1897) (2014)
