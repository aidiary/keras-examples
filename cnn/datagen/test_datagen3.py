import os
import glob
import shutil
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

OUT_DIR = "preview"
IMAGE_FILE = "../../vgg16/elephant.jpg"

datagen = ImageDataGenerator(rotation_range=180)

# 画像をロード（PIL形式画像）
img = load_img(IMAGE_FILE)

# numpy arrayに変換（row, col, channel)
x = img_to_array(img)
# print(x.shape)

# 4次元テンソルに変換（sample, row, col, channel)
x = np.expand_dims(x, axis=0)
# print(x.shape)

if os.path.exists(OUT_DIR):
    shutil.rmtree(OUT_DIR)

os.mkdir(OUT_DIR)

# xは1サンプルのみなのでbatch_sizeは1で固定
g = datagen.flow(x, batch_size=1, save_to_dir=OUT_DIR, save_prefix='img', save_format='jpg')
for i in range(10):
    batch = g.next()
    # print(batch.shape)

images = glob.glob(os.path.join(OUT_DIR, "*.jpg"))

fig = plt.figure()
gs = gridspec.GridSpec(2, 5)
gs.update(wspace=0.1, hspace=0.1)
for i in range(10):
    img = load_img(images[i])
    plt.subplot(gs[i])
    plt.imshow(img)
    plt.axis("off")
plt.show()
