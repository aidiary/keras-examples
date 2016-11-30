Kerasによるデータ拡張
2016/11/29

今回は、画像認識の精度向上に有効な **データ拡張（Data Augmentation）** を実験してみた。データ拡張は、訓練データの画像に対して移動、回転、拡大・縮小などを人工的な操作を加えることでデータ数を水増しするテクニック。画像の移動、回転、拡大・縮小に対してロバストになるため認識精度が向上するようだ。音声認識でも訓練音声に人工的なノイズを上乗せしてデータを拡張して学習するというテクニックを聞いたことがあるのでそれの画像版みたいなものだろう。

Kerasには画像データの拡張を簡単に行う`ImageDataGenerator`というクラスが用意されている。今回は、この使い方をまとめておきたい。

# ImageDataGenerator

Kerasのドキュメントで調べるとこのクラスにはパラメータが大量にあって目が回る。一気に理解するのは難しいので一つずつ検証しよう。

```python
keras.preprocessing.image.ImageDataGenerator(featurewise_center=False,
    samplewise_center=False,
    featurewise_std_normalization=False,
    samplewise_std_normalization=False,
    zca_whitening=False,
    rotation_range=0.,
    width_shift_range=0.,
    height_shift_range=0.,
    shear_range=0.,
    zoom_range=0.,
    channel_shift_range=0.,
    fill_mode='nearest',
    cval=0.,
    horizontal_flip=False,
    vertical_flip=False,
    rescale=None,
    dim_ordering=K.image_dim_ordering())
```

まずは、適当な画像を入力し、各パラメータでどのような画像が生成されるか確認してみた。

```python
if __name__ == '__main__':
    # 画像をロード（PIL形式画像）
    img = load_img(IMAGE_FILE)

    # numpy arrayに変換（row, col, channel)
    x = img_to_array(img)
    # print(x.shape)

    # 4次元テンソルに変換（sample, row, col, channel)
    x = np.expand_dims(x, axis=0)
    # print(x.shape)

    # パラメータを一つだけ指定して残りはデフォルト
    datagen = ImageDataGenerator(rotation_range=90)

    # 生成した画像をファイルに保存
    draw_images(datagen, x, "result_rotation.jpg")
```

`draw_images()`は自作の関数でジェネレータ、画像の4次元テンソル、出力ファイル名を指定すると生成した画像をファイルに描画する。

```python
def draw_images(datagen, x, result_images):
    # 出力先ディレクトリを作成
    temp_dir = "temp"
    os.mkdir(temp_dir)

    # generatorから9個の画像を生成
    # xは1サンプルのみなのでbatch_sizeは1で固定
    g = datagen.flow(x, batch_size=1, save_to_dir=temp_dir, save_prefix='img', save_format='jpg')
    for i in range(9):
        batch = g.next()

    # 生成した画像を3x3で描画
    images = glob.glob(os.path.join(temp_dir, "*.jpg"))
    fig = plt.figure()
    gs = gridspec.GridSpec(3, 3)
    gs.update(wspace=0.1, hspace=0.1)
    for i in range(9):
        img = load_img(images[i])
        plt.subplot(gs[i])
        plt.imshow(img, aspect='auto')
        plt.axis("off")
    plt.savefig(result_images)

    # 出力先ディレクトリを削除
    shutil.rmtree(temp_dir)
```

早速やってみよう。入力画像はなんでもいいけど下のを入れた。

## rotation_range

画像を指定角度の範囲でランダムに回転する。

```python
datagen = ImageDataGenerator(rotation_range=90)
draw_images(datagen, x, "result_rotation.jpg")
```

【画像】

## width_shift_range

画像を指定範囲（画像の横幅に対する割合で指定）でランダムに水平移動する。

```python
datagen = ImageDataGenerator(width_shift_range=0.2)
draw_images(datagen, x, "result_width_shift.jpg")
```

## height_shift_range

```python
datagen = ImageDataGenerator(height_shift_range=0.2)
draw_images(datagen, x, "result_height_shift.jpg")
```

## shear_range

```python
datagen = ImageDataGenerator(shear_range=0.78)  # pi/4
draw_images(datagen, x, "result_shear.jpg")
```

## zoom_range

```python
datagen = ImageDataGenerator(zoom_range=0.5)
draw_images(datagen, x, "result_zoom.jpg")
```

## channel_shift_range

```python
datagen = ImageDataGenerator(channel_shift_range=100)
draw_images(datagen, x, "result_channel_shift.jpg")
```

## horizontal_flip

```python
datagen = ImageDataGenerator(horizontal_flip=True)
draw_images(datagen, x, "result_horizontal_flip.jpg")
```

## vertical_flip

```python
datagen = ImageDataGenerator(vertical_flip=True)
draw_images(datagen, x, "result_vertical_flip.jpg")
```

## samplewise_center

```python
datagen = ImageDataGenerator(samplewise_center=True)
draw_images(datagen, x, "result_samplewise_center.jpg")
```

## samplewise_std_normalization

```python
datagen = ImageDataGenerator(samplewise_std_normalization=True)
draw_images(datagen, x, "result_samplewise_std_normalization.jpg")
```

# ZCA白色化


```python
if __name__ == '__main__':
    img_rows, img_cols, img_channels = 32, 32, 3
    batch_size = 16
    nb_classes = 10

    # CIFAR-10データをロード
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()

    # 画素値を0-1に変換
    X_train = X_train.astype('float32')
    X_train /= 255.0

    draw(X_train[0:batch_size], 'original.png')

    # データ拡張
    datagen = ImageDataGenerator(zca_whitening=True)

    datagen.fit(X_train)
    g = datagen.flow(X_train, y_train, batch_size, shuffle=False)
    X_batch, y_batch = g.next()
    # print(X_batch.shape)
    # print(y_batch.shape)

    draw(X_batch, 'result_zca_whitening.png')
```
