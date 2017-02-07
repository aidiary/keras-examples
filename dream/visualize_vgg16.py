from vgg16_linear import VGG16
from keras.layers import Input
from keras import backend as K
import numpy as np
import matplotlib.pyplot as plt
import json

img_width, img_height, num_channels = 224, 224, 3

input_tensor = Input(shape=(img_height, img_width, num_channels))
model = VGG16(include_top=True, weights='imagenet', input_tensor=input_tensor)
layer_dict = dict([(layer.name, layer) for layer in model.layers])
model.summary()


def deprocess_image(x):
    # テンソルを平均0、標準偏差0.1になるように正規化
    x -= x.mean()
    x /= (x.std() + 1e-5)
    x *= 0.1

    # [0, 1]にクリッピング
    x += 0.5
    x = np.clip(x, 0, 1)

    # RGBに変換
    x *= 255
    x = np.clip(x, 0, 255).astype('uint8')

    return x


def normalize(img, value):
    return value / np.prod(K.int_shape(img)[1:])


def rmsprop(grads, cache=None, decay_rate=0.95):
    if cache is None:
        cache = np.zeros_like(grads)
    cache = decay_rate * cache + (1 - decay_rate) * grads ** 2
    step = grads / np.sqrt(cache + K.epsilon())

    return step, cache


def visualize_filter(layer_name, filter_index, num_loops=200):
    if layer_name not in layer_dict:
        print("ERROR: invalid layer name: %s" % layer_name)
        return

    # 指定した層
    layer = layer_dict[layer_name]

    # layer.output_shape[-1]はどの層でもフィルタ数にあたる（tfの場合）
    # predictions層の場合はクラス数になる
    if not (0 <= filter_index < layer.output_shape[-1]):
        print("ERROR: invalid filter index: %d" % filter_index)
        return

    # 指定した層の指定したフィルタの出力の平均
    # TODO: 損失は通常最小化を目指すものなので別の名前か負記号をつけた方がよい？
    activation_weight = 1.0
    if layer_name == 'predictions':
        loss = activation_weight * K.mean(layer.output[:, filter_index])
    else:
        loss = activation_weight * K.mean(layer.output[:, :, :, filter_index])

    # Lp正則化項
    # 今回の設定ではlossは大きい方がよいためペナルティ項は差し引く
    p = 6.0
    lpnorm_weight = 10.0
    if np.isinf(p):
        lp = K.max(input_tensor)
    else:
        lp = K.pow(K.sum(K.pow(K.abs(input_tensor), p)), 1.0 / p)
    loss -= lpnorm_weight * normalize(input_tensor, lp)

    # Total Variationによる正則化
    # 今回の設定ではlossは大きい方がよいためペナルティ項は差し引く
    beta = 2.0
    tv_weight = 10.0
    a = K.square(input_tensor[:, 1:, :-1, :] - input_tensor[:, :-1, :-1, :])
    b = K.square(input_tensor[:, :-1, 1:, :] - input_tensor[:, :-1, :-1, :])
    tv = K.sum(K.pow(a + b, beta / 2.0))
    loss -= tv_weight * normalize(input_tensor, tv)

    # 層の出力の入力画像に対する勾配を求める
    # 入力画像を微小量変化させたときの出力の変化量を意味する
    # 層の出力を最大化したいためこの勾配を画像に足し込む
    grads = K.gradients(loss, input_tensor)[0]

    # 正規化トリック
    # 画像に勾配を足し込んだときにちょうどよい値になる
    grads /= (K.sqrt(K.mean(K.square(grads))) + K.epsilon())

    # 画像を入力して層の出力と勾配を返す関数を定義
    iterate = K.function([input_tensor], [loss, grads])

    # ノイズを含んだ画像（4Dテンソル）から開始する
    x = np.random.random((1, img_height, img_width, 3))
    x = (x - 0.5) * 20 + 128

    # 初期画像を描画
    # img = deprocess_image(x[0])
    # plt.imshow(img)
    # plt.show()

    # 勾配法で層の出力（loss_value）を最大化するように入力画像を更新する
    cache = None
    for i in range(num_loops):
        loss_value, grads_value = iterate([x])
        # loss_valueを大きくしたいので画像に勾配を加える
        step, cache = rmsprop(grads_value, cache)
        x += step
        print(i, loss_value)

    img = deprocess_image(x[0])

    return img


if __name__ == '__main__':
    # visualize_filter('block1_conv2', 1)
    # visualize_filter('block2_conv2', 29)
    # visualize_filter('block5_conv3', 501)
    # visualize_filter('predictions', 20)  # water_ouzel
    # visualize_filter('predictions', 1)   # goldfish
    # visualize_filter('predictions', 9)   # ostrich
    # visualize_filter('predictions', 130) # flamingo

    class_index = json.load(open('imagenet_class_index.json'))

    # ランダムにクラスを選択
    num_images = 16
    np.random.seed(0)
    target_index = [np.random.randint(0, 1000) for x in range(num_images)]

    # 4x4で描画
    fig = plt.figure(figsize=(8, 8))
    ax = [fig.add_subplot(4, 4, i + 1) for i in range(num_images)]
    for i, a in enumerate(ax):
        a.imshow(visualize_filter('predictions', target_index[i], num_loops=1000))
        a.get_xaxis().set_visible(False)
        a.get_yaxis().set_visible(False)
        a.set_aspect('equal')

        # クラス名を画像に描画
        a.text(5, 20, class_index['%d' % target_index[i]][1])

    fig.subplots_adjust(wspace=0, hspace=0)
    fig.tight_layout()
    fig.savefig('result.png')
