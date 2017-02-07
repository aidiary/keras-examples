from keras.applications.vgg16 import VGG16
from keras.layers import Input
from keras import backend as K
import numpy as np
import matplotlib.pyplot as plt

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


def visualize_filter(layer_name, filter_index):
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
    if layer_name == 'predictions':
        loss = K.mean(layer.output[:, filter_index])
    else:
        loss = K.mean(layer.output[:, :, :, filter_index])

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

    # 勾配法で損失を小さくするように入力画像を更新する
    weight = 1.0
    for i in range(200):
        loss_value, grads_value = iterate([x])
        # loss_valueを大きくしたいので画像に勾配を加える
        x += weight * grads_value
        print(i, loss_value)

    img = deprocess_image(x[0])
    plt.imshow(img)
    plt.show()


if __name__ == '__main__':
    # visualize_filter('block1_conv1', 0)
    # visualize_filter('block5_conv3', 501)
    visualize_filter('predictions', 20)
    visualize_filter('predictions', 64)
