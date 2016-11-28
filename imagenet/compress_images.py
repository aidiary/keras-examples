import os
import string

IMAGE_DIR = "images"

for ch in string.ascii_lowercase:
    command = "tar cvzf images_%s.tar.gz %s/{%s,%s}*/*" \
        % (ch, IMAGE_DIR, ch, ch.upper())
    print(command)
    os.system(command)
