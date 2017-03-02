import numpy as np
import matplotlib.pyplot as plt
from keras.applications.inception_v3 import InceptionV3

def draw_image(img):
    fig, ax = plt.subplots()
    ax.imshow(img)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    plt.show()


# ランダム画像から始める
img_noise = np.random.uniform(size=(224, 224, 3)) + 100.0
# draw_image(img_noise)


def render_naive(target_output, img0=img_noise, iter_n=20, step=1.0):
    print(target_output)





model = InceptionV3(weights='imagenet', include_top=True)
layer_dict = dict([(layer.name, layer) for layer in model.layers])
print(len(layer_dict))


target_layer = 'convolution2d_40'
target_channel = 139
