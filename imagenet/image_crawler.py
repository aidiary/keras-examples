import os
import time
import json
import random
import imghdr
import requests

IMAGE_URL_API = 'http://www.image-net.org/api/text/imagenet.synset.geturls?wnid='

OUTPUT_DIR = "images"
MAX_NUM_IMAGES_PER_CATEGORY = 100


def download_image(url, filename):
    try:
        r = requests.get(url)
    except Exception:
        return False

    if not r.ok:
        return False

    imagetype = imghdr.what(None, h=r.content)
    if imagetype != "jpeg":
        return False

    with open(filename, 'wb') as fp:
        fp.write(r.content)

    # flickr error image
    if os.path.getsize(filename) == 2051:
        os.remove(filename)
        return False

    return True


if __name__ == '__main__':
    if not os.path.exists(OUTPUT_DIR):
        os.mkdir(OUTPUT_DIR)

    # 画像を収集したいカテゴリのリストを読み込む
    # ISVRC2014の1000カテゴリ
    # http://image-net.org/challenges/LSVRC/2014/browse-synsets
    # https://github.com/fchollet/keras/blob/master/keras/applications/imagenet_utils.py
    with open('imagenet_class_index.json', 'r') as fp:
        class_list = json.load(fp)

    # 各カテゴリについて画像を収集
    for wnid, category in class_list.values():
        print("*** category = %s (%s)" % (category, wnid))

        # すでに画像ディレクトリがあったら収集済みなのでスキップする
        if os.path.exists(os.path.join(OUTPUT_DIR, category)):
            print("SKIP")
            continue

        # wnidに属する画像のURLリストをAPIで取得する
        r = requests.get(IMAGE_URL_API + wnid)
        if not r.ok:
            print("WARNING: cannot get image list: wnid = %s" % wnid)
            continue

        page = r.text
        image_url_list = page.rstrip().split('\r\n')
        random.shuffle(image_url_list)

        os.mkdir(os.path.join(OUTPUT_DIR, category))

        num_ok = 0
        for image_url in image_url_list:
            try:
                print("%s ... " % image_url, end='')

                filename = image_url.split('/')[-1]
                ret = download_image(image_url, os.path.join(OUTPUT_DIR, category, filename))

                if ret:
                    print("OK")
                    num_ok += 1
                    if num_ok == MAX_NUM_IMAGES_PER_CATEGORY:
                        break
                else:
                    print("NG")

                # 同じドメインが連続する場合もあるので適宜スリープ
                time.sleep(3)
            except Exception:
                continue
