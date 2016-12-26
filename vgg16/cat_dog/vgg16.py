from keras.applications.vgg16 import VGG16
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import optimizers
import numpy as np
from cat_dog import save_history


img_width, img_height = 150, 150
train_data_dir = 'data/train'
validation_data_dir = 'data/validation'
nb_train_samples = 2000
nb_validation_samples = 800
nb_epoch = 50
top_model_weights_path = 'bottleneck_fc_model.h5'


def save_bottleneck_features():
    """VGG16にDog vs Catの訓練画像、バリデーション画像を入力し、
    ボトルネック特徴量（FC層の直前の出力）をファイルに保存する"""

    # VGG16モデルと学習済み重みをロード
    # Fully-connected層（FC）はいらないのでinclude_top=False）
    model = VGG16(include_top=False, weights='imagenet')
    model.summary()

    # ジェネレータの設定
    datagen = ImageDataGenerator(rescale=1.0 / 255)

    # Dog vs Catのトレーニングセットを生成するジェネレータを作成
    generator = datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        batch_size=32,
        class_mode=None,
        shuffle=False)

    # ジェネレータから生成される画像を入力し、VGG16の出力をファイルに保存
    bottleneck_features_train = model.predict_generator(generator, nb_train_samples)
    np.save('bottleneck_features_train.npy',
            bottleneck_features_train)

    # Dog vs Catのバリデーションセットを生成するジェネレータを作成
    generator = datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_width, img_height),
        batch_size=32,
        class_mode=None,
        shuffle=False)

    # ジェネレータから生成される画像を入力し、VGG16の出力をファイルに保存
    bottleneck_features_validation = model.predict_generator(generator, nb_validation_samples)
    np.save('bottleneck_features_validation.npy',
            bottleneck_features_validation)


def train_top_model():
    """VGGのボトルネック特徴量を入力とし、Dog vs Catの正解を出力とするFCネットワークを訓練"""
    # 訓練データをロード
    # ジェネレータではshuffle=Falseなので最初の1000枚がcats、次の1000枚がdogs
    train_data = np.load('bottleneck_features_train.npy')
    train_labels = np.array([0] * int(nb_train_samples / 2) + [1] * int(nb_train_samples / 2))

    # (2000, 4, 4, 512)
    print(train_data.shape)

    # バリデーションデータをロード
    validation_data = np.load('bottleneck_features_validation.npy')
    validation_labels = np.array([0] * int(nb_validation_samples / 2) + [1] * int(nb_validation_samples / 2))

    # (800, 4, 4, 512)
    print(validation_data.shape)

    # FCネットワークを構築
    model = Sequential()
    model.add(Flatten(input_shape=train_data.shape[1:]))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy',
                  optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
                  metrics=['accuracy'])

    history = model.fit(train_data, train_labels,
                        nb_epoch=nb_epoch, batch_size=32,
                        validation_data=(validation_data, validation_labels))

    model.save_weights(top_model_weights_path)
    save_history(history, 'history_vgg16.txt')


if __name__ == '__main__':
    save_bottleneck_features()
    train_top_model()
