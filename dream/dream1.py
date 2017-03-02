import numpy as np
import matplotlib.pyplot as plt
from keras import backend as K
from keras.layers import Input
from keras.applications.inception_v3 import InceptionV3


# ランダム画像から始める
img_noise = np.random.uniform(size=(1, 299, 299, 3)) + 100.0
# draw_image(img_noise)

# input_tensorはバッチを入れない3Dテンソル
input_tensor = Input(shape=(299, 299, 3))
model = InceptionV3(include_top=True, weights='imagenet', input_tensor=input_tensor)
layer_dict = dict([(layer.name, layer) for layer in model.layers])


def draw_image(img):
    fig, ax = plt.subplots()
    ax.imshow(img)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    plt.show()


def render_naive(layer_name, filter_index, img0=img_noise, iter_n=20, step=1.0):
    if layer_name not in layer_dict:
        print("ERROR: invalid layer name: %s" % layer_name)
        return

    layer = layer_dict[layer_name]

    print("{} < {}".format(filter_index, layer.output_shape[-1]))

    activation = K.mean(layer.output[:, :, :, filter_index])
    grads = K.gradients(activation, input_tensor)[0]

    # DropoutやBNを含むネットワークはK.learning_phase()が必要
    iterate = K.function([input_tensor, K.learning_phase()], [activation, grads])

    img = img0.copy()
    for i in range(iter_n):
        # 学習はしないので0を入力
        activation_value, grads_value = iterate([img, 0])
        grads_value /= K.std(grads_value) + 1e-8
        img += grads_value * step
        print(i, activation_value)

render_naive('convolution2d_40', 139)
