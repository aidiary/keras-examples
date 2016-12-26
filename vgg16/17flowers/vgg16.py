from keras.applications.vgg16 import VGG16
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import optimizers
from keras.utils import np_utils
import numpy as np
from smallcnn import save_history


classes = ['Tulip', 'Snowdrop', 'LilyValley', 'Bluebell', 'Crocus',
           'Iris', 'Tigerlily', 'Daffodil', 'Fritillary', 'Sunflower',
           'Daisy', 'ColtsFoot', 'Dandelion', 'Cowslip', 'Buttercup',
           'Windflower', 'Pansy']

batch_size = 32
nb_classes = len(classes)

img_rows, img_cols = 150, 150
channels = 3

train_data_dir = 'train_images'
validation_data_dir = 'test_images'

nb_samples_per_class = 70

nb_train_samples = 1190
nb_val_samples = 170
nb_epoch = 100

top_model_weights_path = 'bottleneck_fc_model.h5'


def save_bottleneck_features():
    """VGG16に訓練画像、バリデーション画像を入力し、
    ボトルネック特徴量（FC層の直前の出力）をファイルに保存する"""

    # VGG16モデルと学習済み重みをロード
    # Fully-connected層（FC）はいらないのでinclude_top=False）
    model = VGG16(include_top=False, weights='imagenet')
    model.summary()

    # ジェネレータの設定
    datagen = ImageDataGenerator(rescale=1.0 / 255)

    # 訓練セットを生成するジェネレータを作成
    generator = datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_rows, img_cols),
        color_mode='rgb',
        classes=classes,
        class_mode='categorical',
        batch_size=batch_size,
        shuffle=False)

    # ジェネレータから生成される画像を入力し、VGG16の出力をファイルに保存
    bottleneck_features_train = model.predict_generator(generator, nb_train_samples)
    np.save('bottleneck_features_train.npy', bottleneck_features_train)

    # バリデーションセットを生成するジェネレータを作成
    generator = datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_rows, img_cols),
        color_mode='rgb',
        classes=classes,
        class_mode='categorical',
        batch_size=batch_size,
        shuffle=False)

    # ジェネレータから生成される画像を入力し、VGG16の出力をファイルに保存
    bottleneck_features_validation = model.predict_generator(generator, nb_val_samples)
    np.save('bottleneck_features_validation.npy', bottleneck_features_validation)


def train_top_model():
    """VGGのボトルネック特徴量を入力とし、正解を出力とするFCネットワークを訓練"""
    # 訓練データをロード
    # ジェネレータではshuffle=Falseなのでクラスは順番に出てくる
    # one-hot vector表現へ変換が必要
    train_data = np.load('bottleneck_features_train.npy')
    train_labels = [i // nb_samples_per_class for i in range(nb_train_samples)]
    train_labels = np_utils.to_categorical(train_labels, nb_classes)

    # バリデーションデータをロード
    validation_data = np.load('bottleneck_features_validation.npy')
    validation_labels = [i // nb_samples_per_class for i in range(nb_val_samples)]
    validation_labels = np_utils.to_categorical(validation_labels, nb_classes)

    # FCネットワークを構築
    model = Sequential()
    model.add(Flatten(input_shape=train_data.shape[1:]))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(nb_classes, activation='softmax'))

    model.summary()

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    history = model.fit(train_data, train_labels,
                        nb_epoch=nb_epoch, batch_size=batch_size,
                        validation_data=(validation_data, validation_labels))

    model.save_weights(top_model_weights_path)
    save_history(history, 'results/history_vgg16.txt')


if __name__ == '__main__':
    # save_bottleneck_features()
    train_top_model()
