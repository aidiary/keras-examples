import os
import shutil

IN_DIR = 'jpg'
OUT_DIR = 'images'
if not os.path.exists(OUT_DIR):
    os.mkdir(OUT_DIR)

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
    os.mkdir(os.path.join(OUT_DIR, name))

# jpgをスキャン
for f in sorted(os.listdir(IN_DIR)):
    # image_0001.jpg => 1
    prefix = f.replace('.jpg', '')
    idx = int(prefix.split('_')[1])

    for name in flower_dics:
        start, end = flower_dics[name]
        if idx in range(start, end + 1):
            source = os.path.join(IN_DIR, f)
            dest = os.path.join(OUT_DIR, name)
            shutil.copy(source, dest)
            continue
