VGG16のFine-tuningによる花の分類
2017/1/17

前回は、VGG16をFine-tuningして犬か猫を分類できるニューラルネットワークを学習した。今回は、同様のアプローチで17種類の花を分類するニューラルネットワークを学習してみたい。前回の応用編みたいな感じ。

使用したデータは、VGG16を提案したOxford大学のグループが公開している [17 Category Flower Dataset](http://www.robots.ox.ac.uk/~vgg/data/flowers/17/) である。下のような17種類の花の画像データ。

【画像】

前に試したようにデフォルトのVGG16ではひまわりさえ分類できなかったが、VGG16をFine-tuningすることで果たしてこれら17種類の花分類できるようになるのだろうか？わくわく。

# セットアップ

`17flowers.tgz`というファイルをダウンロードする。これを解凍すると`jpg`というディレクトリの中に`image_0001.jpg`から`image_1360.jpg`まで1360枚の画像がある。1360枚は畳み込みニューラルネットワークをスクラッチから学習するには心もとないデータ数であるが、今回はVGG16を少量データでチューニングする転移学習を使うので十分だろう。

各クラス80枚で17クラスなので1360枚なのだが、各画像がどのラベルなのかがわからない。とりあえずサンプル画像と見比べて下のようにラベルを割り振った`labels.txt`というファイルを作成した。たとえば、1行目は`image_0001.jpg`から`image_0080.jpg`までが`Tulip`であることを意味する。

```
1       80      Tulip
81      160     Snowdrop
161     240     LilyValley
241     320     Bluebell
321     400     Crocus
401     480     Iris
481     560     Tigerlily
561     640     Daffodil
641     720     Fritillary
721     800     Sunflower
801     880     Daisy
881     960     ColtsFoot
961     1040    Dandelion
1041    1120    Cowslip
1121    1200    Buttercup
1201    1280    Windflower
1281    1360    Pansy
```

まずは、Kerasからロードしやすいように以下の`setup.py`で画像ファイルを分割する。各クラスで訓練データが70枚、テストデータが10枚になるように分割した。犬猫分類でやったようにサブディレクトリにクラス名を付けておくと自動的に認識してくれる。このクラス名を付けるために先ほどの`labels.txt`を使った。

```python
import os
import shutil
import random

IN_DIR = 'jpg'
TRAIN_DIR = 'train_images'
TEST_DIR = 'test_images'

if not os.path.exists(TRAIN_DIR):
    os.mkdir(TRAIN_DIR)

if not os.path.exists(TEST_DIR):
    os.mkdir(TEST_DIR)

# name => (start idx, end idx)
flower_dics = {}

with open('labels.txt') as fp:
    for line in fp:
        line = line.rstrip()
        cols = line.split()

        assert len(cols) == 3

        start = int(cols[0])
        end = int(cols[1])
        name = cols[2]

        flower_dics[name] = (start, end)

# 花ごとのディレクトリを作成
for name in flower_dics:
    os.mkdir(os.path.join(TRAIN_DIR, name))
    os.mkdir(os.path.join(TEST_DIR, name))

# jpgをスキャン
for f in sorted(os.listdir(IN_DIR)):
    # image_0001.jpg => 1
    prefix = f.replace('.jpg', '')
    idx = int(prefix.split('_')[1])

    for name in flower_dics:
        start, end = flower_dics[name]
        if idx in range(start, end + 1):
            source = os.path.join(IN_DIR, f)
            dest = os.path.join(TRAIN_DIR, name)
            shutil.copy(source, dest)
            continue

# 訓練データの各ディレクトリからランダムに10枚をテストとする
for d in os.listdir(TRAIN_DIR):
    files = os.listdir(os.path.join(TRAIN_DIR, d))
    random.shuffle(files)
    for f in files[:10]:
        source = os.path.join(TRAIN_DIR, d, f)
        dest = os.path.join(TEST_DIR, d)
        shutil.move(source, dest)
```

# 1. 小さな畳み込みニューラルネットをスクラッチから学習する

前回と同様に今回もベースラインとしてLeNet相当の小さな畳み込みニューラルネットワークを学習してみよう。まずはモデルを構築する。

```python
model = Sequential()
model.add(Convolution2D(32, 3, 3, input_shape=(img_rows, img_cols, channels)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Convolution2D(32, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Convolution2D(64, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(nb_classes))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
```

今まで何回もやってきたので特に難しいところはない。今回は17クラスの多クラス分類なので損失関数に`categorical_crossentropy`を使う。

画像ファイルしか提供されていないときはデータの読み込みにImageDataGeneratorを使うと便利。今回もデータ拡張（）は前回と同じく`shear_range`と`zoom_range`と`horizontal_flip`を使ったがデータの特徴を見て慎重に決めるとより精度が向上するかも。

```python
# ディレクトリの画像を使ったジェネレータ
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1.0 / 255)
```

上の設定を使って実際にジェネレータを作成。画像ファイルを含むディレクトリを指定するだけでよい。ここで、`classes`を自分で指定すると順番にクラスラベルを割り振ってくれる。`color_mode`と`class_mode`は何に使われるかいまいち把握できていないが`rgb`と`categorical`でよさそう。

```python
classes = ['Tulip', 'Snowdrop', 'LilyValley', 'Bluebell', 'Crocus',
           'Iris', 'Tigerlily', 'Daffodil', 'Fritillary', 'Sunflower',
           'Daisy', 'ColtsFoot', 'Dandelion', 'Cowslip', 'Buttercup',
           'Windflower', 'Pansy']

train_generator = train_datagen.flow_from_directory(
    directory='train_images',
    target_size=(img_rows, img_cols),
    color_mode='rgb',
    classes=classes,
    class_mode='categorical',
    batch_size=batch_size,
    shuffle=True)

test_generator = test_datagen.flow_from_directory(
    directory='test_images',
    target_size=(img_rows, img_cols),
    color_mode='rgb',
    classes=classes,
    class_mode='categorical',
    batch_size=batch_size,
    shuffle=True)
```

ジェネレータができたのでジェネレータが生成する画像の4Dテンソルを使ってモデルを訓練する。ここは前回と同じ。

```python
history = model.fit_generator(
    train_generator,
    samples_per_epoch=samples_per_epoch,
    nb_epoch=nb_epoch,
    validation_data=test_generator,
    nb_val_samples=nb_val_samples)
save_history(history, os.path.join(result_dir, 'history_smallcnn.txt'))
```

学習途中の損失と精度はあとで参照できるようにファイルに保存した。


# 2. VGG16をスクラッチから学習する




# 3. VGG16をFine-tuningする

最後にVGG16をFine-tuningしてみよう。

```python
# VGG16モデルと学習済み重みをロード
# Fully-connected層（FC）はいらないのでinclude_top=False）
input_tensor = Input(shape=(img_rows, img_cols, 3))
vgg16 = VGG16(include_top=False, weights='imagenet', input_tensor=input_tensor)

# FC層を構築
top_model = Sequential()
top_model.add(Flatten(input_shape=vgg16.output_shape[1:]))
top_model.add(Dense(256, activation='relu'))
top_model.add(Dropout(0.5))
top_model.add(Dense(nb_classes, activation='softmax'))

# VGG16とFCを接続
model = Model(input=vgg16.input, output=top_model(vgg16.output))

# 最後のconv層の直前までの層をfreeze
for layer in model.layers[:15]:
    layer.trainable = False

# Fine-tuningのときはSGDの方がよい？
model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
              metrics=['accuracy'])
```

前回とほとんど同じ。違いは、FC層の出力が17クラス分類になるので活性化関数に`softmax`を使っているところと損失関数に`categorical_crossentropy`を使っているところくらい。

あとはほとんど同じなので省略。

# 実験結果

# 花の分類例

# 参考
