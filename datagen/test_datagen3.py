import os
import glob
import shutil
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

IMAGE_FILE = "../vgg16/elephant.jpg"


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


if __name__ == '__main__':
    # 画像をロード（PIL形式画像）
    img = load_img(IMAGE_FILE)

    # numpy arrayに変換（row, col, channel)
    x = img_to_array(img)
    # print(x.shape)

    # 4次元テンソルに変換（sample, row, col, channel)
    x = np.expand_dims(x, axis=0)
    # print(x.shape)

    datagen = ImageDataGenerator(rotation_range=90)
    draw_images(datagen, x, "result_rotation.jpg")

    datagen = ImageDataGenerator(width_shift_range=0.2)
    draw_images(datagen, x, "result_width_shift.jpg")

    datagen = ImageDataGenerator(height_shift_range=0.2)
    draw_images(datagen, x, "result_height_shift.jpg")

    datagen = ImageDataGenerator(shear_range=0.78)  # pi/4
    draw_images(datagen, x, "result_shear.jpg")

    datagen = ImageDataGenerator(zoom_range=0.5)
    draw_images(datagen, x, "result_zoom.jpg")

    datagen = ImageDataGenerator(channel_shift_range=100)
    draw_images(datagen, x, "result_channel_shift.jpg")

    datagen = ImageDataGenerator(horizontal_flip=True)
    draw_images(datagen, x, "result_horizontal_flip.jpg")

    datagen = ImageDataGenerator(vertical_flip=True)
    draw_images(datagen, x, "result_vertical_flip.jpg")

    datagen = ImageDataGenerator(samplewise_center=True)
    draw_images(datagen, x, "result_samplewise_center.jpg")

    datagen = ImageDataGenerator(samplewise_std_normalization=True)
    draw_images(datagen, x, "result_samplewise_std_normalization.jpg")
