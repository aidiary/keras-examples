import os
import sys
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.models import Sequential, Model
from keras.layers import Input, Activation, Dropout, Flatten, Dense
from keras.preprocessing import image
import numpy as np

if len(sys.argv) != 2:
    print("usage: python predict.py [filename]")
    sys.exit(1)

filename = sys.argv[1]
print('input:', filename)

result_dir = 'results'

classes = ['Tulip', 'Snowdrop', 'LilyValley', 'Bluebell', 'Crocus',
           'Iris', 'Tigerlily', 'Daffodil', 'Fritillary', 'Sunflower',
           'Daisy', 'ColtsFoot', 'Dandelion', 'Cowslip', 'Buttercup',
           'Windflower', 'Pansy']
nb_classes = len(classes)

img_height, img_width = 150, 150
channels = 3

# VGG16
input_tensor = Input(shape=(img_height, img_width, channels))
vgg16 = VGG16(include_top=False, weights='imagenet', input_tensor=input_tensor)

# FC
fc = Sequential()
fc.add(Flatten(input_shape=vgg16.output_shape[1:]))
fc.add(Dense(256, activation='relu'))
fc.add(Dropout(0.5))
fc.add(Dense(nb_classes, activation='softmax'))

# VGG16とFCを接続
model = Model(input=vgg16.input, output=fc(vgg16.output))

# 学習済みの重みをロード
model.load_weights(os.path.join(result_dir, 'finetuning.h5'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
# model.summary()

# 画像を読み込んで4次元テンソルへ変換
img = image.load_img(filename, target_size=(img_height, img_width))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)

x = preprocess_input(x)

# print(x)
# print(x.shape)

# クラスを予測
# 入力は1枚の画像なので[0]のみ
pred = model.predict(x)[0]

# 予測確率が高いトップ5を出力
top = 5
top_indices = pred.argsort()[-top:][::-1]
result = [(classes[i], pred[i]) for i in top_indices]
for x in result:
    print(x)
