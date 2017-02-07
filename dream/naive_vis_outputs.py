from keras.applications.vgg16 import VGG16
from keras.layers import Input
from keras import backend as K
import numpy as np
import matplotlib.pyplot as plt
from scipy.misc import toimage

def deprocess_image(x):
    # normalize tensor: center on 0., ensure std is 0.1
    x -= x.mean()
    x /= (x.std() + 1e-5)
    x *= 0.1

    # clip to [0, 1]
    x += 0.5
    x = np.clip(x, 0, 1)

    # convert to RGB array
    x *= 255
    x = np.clip(x, 0, 255).astype('uint8')

    return x

img_width, img_height = 224, 224

input_tensor = Input(shape=(img_height, img_width, 3))
vgg16 = VGG16(include_top=True, weights='imagenet', input_tensor=input_tensor)
vgg16.summary()

layer_dict = dict([(layer.name, layer) for layer in vgg16.layers])

# 可視化したいクラスのインデックスを指定
class_index = 999

# クラス出力
layer_output = vgg16.layers[-1].output

# 指定した層の指定したフィルタの出力の平均を出力とする
# この出力を最大化する入力画像を勾配法で求める
target_output = K.mean(layer_output[0, class_index])

# 出力の入力画像に対する勾配を計算
# 入力画像を微少量変化させたときの出力の変化量を意味する
grads = K.gradients(target_output, input_tensor)[0]

# 正規化トリック（stepを大きくすれば不要）
#grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-5)

# ループ関数を定義
iterate = K.function([input_tensor], [target_output, grads])

# ノイズを含んだグレーイメージから開始
input_img_data = np.random.randint(-20, 20, (1, img_height, img_width, 3)) + 128
input_img_data = input_img_data.astype(np.float64)

# 勾配法で入力画像を求める
step = 10000.0
for i in range(1000):
    target_output_value, grads_value = iterate([input_img_data])
    # 出力を大きくする方に動かしたいので入力画像に勾配を足し込む
    # stepは学習率に当たる
    input_img_data += grads_value * step
    print(i, target_output_value)

    # もし出力確率が99%を超えたらループ終了
    if target_output_value > 0.9:
        break

img = deprocess_image(input_img_data[0])
plt.imshow(img)
plt.show()
