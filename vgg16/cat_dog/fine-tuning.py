from keras.applications.vgg16 import VGG16
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, Model
from keras.layers import Input, Activation, Dropout, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
import numpy as np
from cat_dog import save_history

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
    # input_tensorを指定しておかないとoutput_shapeがNoneになってエラーになるので注意
    # https://keras.io/applications/#inceptionv3
    input_tensor = Input(shape=(img_height, img_width, 3))
    vgg16_model = VGG16(include_top=False, weights='imagenet', input_tensor=input_tensor)
    # vgg16_model.summary()

    # FC層を構築
    # Flattenへの入力指定はバッチ数を除く
    top_model = Sequential()
    top_model.add(Flatten(input_shape=vgg16_model.output_shape[1:]))
    top_model.add(Dense(256, activation='relu'))
    top_model.add(Dropout(0.5))
    top_model.add(Dense(1, activation='sigmoid'))

    # 学習済みのFC層の重みをロード
    # TODO: 要検証：ランダムな重みではうまくいかないようだ
    top_model.load_weights(top_model_weights_path)

    # vgg16_modelはkeras.engine.training.Model
    # top_modelはSequentialとなっている
    # ModelはSequentialでないためadd()がない
    # そのためFunctional APIで二つのモデルを結合する
    # https://github.com/fchollet/keras/issues/4040
    model = Model(input=vgg16_model.input, output=top_model(vgg16_model.output))
    print('vgg16_model:', vgg16_model)
    print('top_model:', top_model)
    print('model:', model)

    # Total params: 16,812,353
    # Trainable params: 16,812,353
    # Non-trainable params: 0
    model.summary()

    # layerを表示
    for i in range(len(model.layers)):
        print(i, model.layers[i])

    # 最後のconv層の直前までの層をfreeze
    for layer in model.layers[:15]:
        layer.trainable = False

    # Total params: 16,812,353
    # Trainable params: 9,177,089
    # Non-trainable params: 7,635,264
    model.summary()

    # TODO: ここでAdamを使うとうまくいかない
    # Fine-tuningのときはSGDの方がよい？
    model.compile(loss='binary_crossentropy',
                  optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
                  metrics=['accuracy'])

    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

    test_datagen = ImageDataGenerator(rescale=1.0 / 255)

    train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_height, img_width),
        batch_size=32,
        class_mode='binary')

    validation_generator = test_datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_height, img_width),
        batch_size=32,
        class_mode='binary')

    # Fine-tuning
    history = model.fit_generator(
        train_generator,
        samples_per_epoch=nb_train_samples,
        nb_epoch=nb_epoch,
        validation_data=validation_generator,
        nb_val_samples=nb_validation_samples)

    model.save_weights('fine-tuning.h5')
    save_history(history, 'history.txt')
