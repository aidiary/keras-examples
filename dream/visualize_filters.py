from keras.applications.vgg16 import VGG16
from keras.layers import Input
from keras import backend as K
import numpy as np
import matplotlib.pyplot as plt
from scipy.misc import toimage

img_width, img_height = 128, 128

input_tensor = Input(shape=(img_height, img_width, 3))
vgg16 = VGG16(include_top=False, weights='imagenet', input_tensor=input_tensor)
vgg16.summary()

print(input_tensor)
print(vgg16.layers[0].input)

layer_dict = dict([(layer.name, layer) for layer in vgg16.layers])
print(layer_dict)

# 可視化したい層の名前を指定
layer_name = 'block5_conv1'
# 層は複数のフィルタからなるため可視化したいフィルタのインデックスを指定
filter_index = 0

# 指定した層の出力
layer_output = layer_dict[layer_name].output

# 指定した層の指定したフィルタの出力の平均を損失とする
# tfなのでchannelは最後の次元
loss = K.mean(layer_output[:, :, :, filter_index])

# 損失に関する入力画像の勾配を計算
grads = K.gradients(loss, input_tensor)[0]

# 正規化トリック
grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-5)

# 訓練用の関数
iterate = K.function([input_tensor], [loss, grads])

# ノイズを含んだグレーイメージから開始
# TODO: 3チャンネルだとグレーにならないのでは？
input_img_data = np.random.random((1, img_height, img_width, 3)) * 20 + 128
img = toimage(input_img_data[0])
plt.imshow(img)
plt.show()

step = 1.0
for i in range(20):
    loss_value, grads_value = iterate([input_img_data])
    input_img_data += grads_value * step
