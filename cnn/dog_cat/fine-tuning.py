from keras.applications.vgg16 import VGG16
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, Model
from keras.layers import Activation, Dropout, Flatten, Dense
import numpy as np
from dog_cat import save_history

# https://gist.github.com/fchollet/7eb39b44eb9e16e59632d25fb3119975

img_width, img_height = 150, 150
train_data_dir = 'data/train'
validation_data_dir = 'data/validation'
nb_train_samples = 2000
nb_validation_samples = 800
nb_epoch = 50

# vgg16.pyで学習済みのFC層の重み
top_model_weights_path = 'bottleneck_fc_model.h5'


if __name__ == '__main__':
    # VGG16モデルと学習済み重みをロード
    # Fully-connected層（FC）はいらないのでinclude_top=False）
    vgg16_model = VGG16(include_top=False, weights='imagenet')
    # vgg16_model.summary()

    # Keras blogではvgg16_model.output_shapeでFC層の直前のshapeが取れるとあるが
    # MaxPooling2Dはパラメータがないので無理みたい？
    # 代わりにその一つ前のConvolution2Dのshapeをとる

    # print(vgg16_model.output_shape)
    # print(vgg16_model.layers[-2].get_weights()[0].shape)

    # FC層を構築
    top_model = Sequential()
    top_model.add(Flatten())
    top_model.add(Dense(256, activation='relu'))
    top_model.add(Dropout(0.5))
    top_model.add(Dense(1, activation='sigmoid'))

    # 学習済みのFC層の重みをロード
    # TODO: 要検証：ランダムな重みではうまくいかないようだ
    top_model.load_weights(top_model_weights_path)

    # print(vgg16_model)
    # print(top_model)
    # print(dir(vgg16_model))

    # VGG16はkeras.engine.training.ModelでSequentialではないためadd()が使えなくなっている
    # Functional APIを使うしかない？
    # https://github.com/fchollet/keras/issues/4040
    # model.add(top_model)
    print(vgg16_model.output)

    model = Model(input=vgg16_model.input, output=top_model(vgg16_model.output))

    print(model)

    model.summary()
