import os
import time
import requests

IMAGE_URL_API = 'http://www.image-net.org/api/text/imagenet.synset.geturls?wnid='

OUTPUT_DIR = "images"
MAX_NUM_IMAGES_PER_CATEGORY = 1

def download_image(url, filename):
    try:
        r = requests.get(url)
    except Exception:
        return False

    if not r.ok:
        return False

    with open(filename, 'wb') as fp:
        fp.write(r.content)

    return True


if __name__ == '__main__':
    if not os.path.exists(OUTPUT_DIR):
        os.mkdir(OUTPUT_DIR)

    # カテゴリ => WordNet IDの辞書を作成
    cat2wnid = dict()
    wnid2cat = dict()
    with open('words.txt', 'r') as fp:
        for line in fp.readlines():
            line = line.rstrip()
            wnid, category = line.split('\t')
            cat2wnid[category] = wnid
            wnid2cat[wnid] = category

    # 画像を収集したいカテゴリのリストを読み込む
    # ISVRC2014の1000カテゴリ
    # http://image-net.org/challenges/LSVRC/2014/browse-synsets
    target_categories = []
    with open('class1000.txt', 'r') as fp:
        for line in fp.readlines():
            line = line.rstrip()
            target_categories.append(line)

    # カテゴリのリストをWordNet IDのリストに変換
    target_wnid = []
    for cat in target_categories:
        target_wnid.append(cat2wnid[cat])

    # 各カテゴリについて画像を収集
    for wnid in target_wnid:
        print("*** wnid = %s (%s)" % (wnid, wnid2cat[wnid]))
        r = requests.get(IMAGE_URL_API + wnid)
        if not r.ok:
            print("WARNING: cannot get image list: wnid = %s" % wnid)
            continue

        page = r.text
        image_url_list = page.rstrip().split('\r\n')

        num_ok = 0
        for image_url in image_url_list:
            print("%s ... " % image_url, end="")

            filename = image_url.split('/')[-1]
            ret = download_image(image_url, os.path.join(OUTPUT_DIR, filename))

            if ret == True:
                print("OK")
                num_ok += 1
                if num_ok == MAX_NUM_IMAGES_PER_CATEGORY:
                    break
            else:
                print("NG")

            # 同じドメインが連続する場合もあるので適宜スリープ
            time.sleep(3)
