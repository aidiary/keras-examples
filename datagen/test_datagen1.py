import os
import shutil
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator

if __name__ == '__main__':
    img_rows, img_cols, img_channels = 32, 32, 3
    batch_size = 32
    nb_classes = 10

    # CIFAR-10データをロード
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()

    # 画素値を0-1に変換
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255.0
    X_test /= 255.0

    # データ拡張
    outdir = 'datagen'

    if os.path.exists(outdir):
        shutil.rmtree(outdir)
    os.mkdir(outdir)

    datagen = ImageDataGenerator(
        zca_whitening=True
    )
    datagen.fit(X_train)
    g = datagen.flow(X_train, y_train, batch_size, save_to_dir=outdir, save_prefix='img', save_format='jpeg')
    batch = g.next()
    print(batch)
