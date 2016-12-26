from keras.applications.vgg16 import VGG16
from keras.models import Sequential, Model
from keras.layers import Input, Activation, Dropout, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator

classes = ['Tulip', 'Snowdrop', 'LilyValley', 'Bluebell', 'Crocus',
           'Iris', 'Tigerlily', 'Daffodil', 'Fritillary', 'Sunflower',
           'Daisy', 'ColtsFoot', 'Dandelion', 'Cowslip', 'Buttercup',
           'Windflower', 'Pansy']

batch_size = 32
nb_classes = len(classes)

img_rows, img_cols = 150, 150
channels = 3

# VGG16
input_tensor = Input(shape=(img_rows, img_cols, channels))
vgg16_model = VGG16(include_top=False, weights='imagenet', input_tensor=input_tensor)

# FC
top_model = Sequential()
top_model.add(Flatten(input_shape=vgg16_model.output_shape[1:]))
top_model.add(Dense(256, activation='relu'))
top_model.add(Dropout(0.5))
top_model.add(Dense(nb_classes, activation='softmax'))

# VGG16とFCを接続
model = Model(input=vgg16_model.input, output=top_model(vgg16_model.output))

# 学習済みの重みをロード
model.load_weights('fine-tuning.h5')
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
model.summary()

# テスト用データを生成するジェネレータ
test_data_dir = 'test_images'
nb_test_samples = 170
test_datagen = ImageDataGenerator(rescale=1.0 / 255)
test_generator = test_datagen.flow_from_directory(
    test_data_dir,
    target_size=(img_rows, img_cols),
    color_mode='rgb',
    classes=classes,
    class_mode='categorical',
    batch_size=batch_size,
    shuffle=True)

loss, acc = model.evaluate_generator(test_generator, val_samples=nb_test_samples)
print(loss, acc)
