from keras.preprocessing.image import load_img, img_to_array
import numpy as np
from scipy.misc import imsave
from scipy.optimize import fmin_l_bfgs_b
import time
import argparse

from keras.applications import vgg16
from keras import backend as K
from keras.layers import Input

parser = argparse.ArgumentParser(description='Deep Dreams with Keras.')
parser.add_argument('base_image_path', metavar='base', type=str,
                    help='Path to the image to transform.')
parser.add_argument('result_prefix', metavar='res_prefix', type=str,
                    help='Prefix for the saved results.')

args = parser.parse_args()
base_image_path = args.base_image_path
result_prefix = args.result_prefix

# 生成される画像の解像度
img_height, img_width = 600, 600

# 面白い結果が得られる設定
saved_settings = {
    'bad_trip': {'features': {'block4_conv1': 0.05,
                              'block4_conv2': 0.01,
                              'block4_conv3': 0.01},
                 'continuity': 0.1,
                 'dream_l2': 0.8,
                 'jitter': 5},
    'dreamy': {'features': {'block5_conv1': 0.05,
                            'block5_conv2': 0.02},
               'continuity': 0.1,
               'dream_l2': 0.02,
               'jitter': 0},
}

settings = saved_settings['dreamy']
print(settings)


def preprocess_image(image_path):
    img = load_img(image_path, target_size=(img_height, img_width))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = vgg16.preprocess_input(img)
    return img


# VGG16の重みはもともとCaffeのものでBGRの画像で学習されている
def deprocess_image(x):
    x = x.reshape((img_height, img_width, 3))

    # VGG16で正規化された平均を足し込む
    x[:, :, 0] += 103.939
    x[:, :, 1] += 116.779
    x[:, :, 2] += 123.68

    # BGR => RGB
    x = x[:, :, ::-1]
    x = np.clip(x, 0, 255).astype('uint8')

    return x


img_size = (img_height, img_width, 3)

# 生成画像を表す
dream = Input(batch_shape=(1, ) + img_size)

# VGG16をロード
model = vgg16.VGG16(input_tensor=dream, weights='imagenet', include_top=False)
print('Model loaded.')

# 層の名前からオブジェクトへの辞書を作成
layer_dict = dict([(layer.name, layer) for layer in model.layers])


def continuity_loss(x):
    assert K.ndim(x) == 4
    a = K.square(x[:, :img_height - 1, :img_width - 1, :] - x[:, 1:, :img_width - 1, :])
    b = K.square(x[:, :img_height - 1, :img_width - 1, :] - x[:, :img_height - 1, 1:, :])
    return K.sum(K.pow(a + b, 1.25))


# 損失を定義
# 層の出力の最大
loss = K.variable(0.0)
for layer_name in settings['features']:
    assert layer_name in layer_dict.keys(), 'Layer ' + layer_name + ' not found in model.'

    coeff = settings['features'][layer_name]

    # 層の出力
    x = layer_dict[layer_name].output
    shape = layer_dict[layer_name].output_shape

    # 層の出力の最大化を目指すので損失関数は負記号を付ける
    # TODO: 画像のスライシングは何を意味する？層の出力のボーダーを避けるための工夫？
    loss -= coeff * K.sum(K.square(x[:, 2:shape[1] - 2, 2:shape[2] - 2, :])) / np.prod(shape[1:])

# lossは最小化を目指すので正則化項は+する
# continuity loss
loss += settings['continuity'] * continuity_loss(dream) / np.prod(img_size)

# L2 norm
loss += settings['dream_l2'] * K.sum(K.square(dream)) / np.prod(img_size)

# 損失関数の入力画像に対する勾配
grads = K.gradients(loss, dream)

# 画像を入力して損失と勾配を返す関数
# TODO: lossとgradsを返すK.functionをわければEvaluator不要では？
f_outputs = K.function([dream], [loss, grads])


# scipyのBFGSには入力xをとるloss(x)とgrads(x)の関数を与える必要がある
# しかし、f_outputsは入力dreamを入れたときのlossとgradsの値を返してしまう
class Evaluator(object):
    def __init__(self):
        self.loss_value = None
        self.grad_value = None

    def loss(self, x):
        """BFGSに与える損失関数"""
        assert self.loss_value is None
        # BFGSのloss()への入力はフラット化されているので4Dテンソルに戻す
        x = x.reshape((1, ) + img_size)

        # loss(x)でlossだけでなくgradもまとめて計算して保存しておく
        loss_value, grad_values = f_outputs([x])

        # grad_valuesは4Dテンソルのままなのでフラット化する
        # astype('float64')がないとfmin_l_bfgs_bが下のエラーをはく
        # ValueError: failed to initialize intent(inout) array -- expected elsize=8 but got 4
        grad_values = grad_values.flatten().astype('float64')

        self.loss_value = loss_value
        self.grad_values = grad_values

        return self.loss_value

    def grads(self, x):
        """BFGSに与える勾配関数"""
        assert self.loss_value is not None

        # loss(x)のf_output()で計算済みのgradをそのまま返す
        grad_values = np.copy(self.grad_values)

        self.loss_value = None
        self.grad_values = None

        return grad_values

evaluator = Evaluator()

# BFGSで損失を最小化する入力画像を求める
x = preprocess_image(base_image_path)
for i in range(5):
    print('Start of iteration', i)

    random_jitter = (settings['jitter'] * 2) * (np.random.random(img_size) - 0.5)
    x += random_jitter

    # print("***", evaluator.loss(x))
    # print("***", evaluator.grads(x).shape)
    # print("***", x.flatten().shape)
    x, min_val, info = fmin_l_bfgs_b(evaluator.loss, x.flatten(), fprime=evaluator.grads, maxfun=7)

    print('Current loss value:', min_val)

    # 画像に戻して保存
    x = x.reshape(img_size)
    x -= random_jitter
    img = deprocess_image(np.copy(x))
    fname = result_prefix + '_at_iteration_%d.png' % i
    imsave(fname, img)
    print('Image saved as', fname)
