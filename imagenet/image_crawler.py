import urllib.request

IMAGE_URL_API = 'http://www.image-net.org/api/text/imagenet.synset.geturls?wnid='

OUTPUT_DIR = "images"

if __name__ == '__main__':
    # カテゴリ => WordNet IDの辞書を作成
    cat2wnid = dict()
    with open('words.txt', 'r') as fp:
        for line in fp.readlines():
            line = line.rstrip()
            wnid, category = line.split('\t')
            cat2wnid[category] = wnid

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
        with urllib.request.urlopen(IMAGE_URL_API + wnid) as res:
            page = res.read().decode('utf-8').rstrip()
            image_url_list = page.split('\r\n')
            print(image_url_list)
        exit()
