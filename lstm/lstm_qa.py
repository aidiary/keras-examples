"""
Facebook bAbiデータを適用したLSTMエンコーダ
https://www.youtube.com/watch?v=YimQOpSRULY&t=885s
https://bitbucket.org/fchollet/keras_workshop
"""

from keras.models import Sequential
from keras.layers.embeddings import Embedding
from keras.layers.core import Activation, Dense, Merge
from keras.layers.recurrent import GRU, LSTM
from keras.utils.data_utils import get_file
import re
import codecs
import numpy as np


def tokenize(sent):
    '''文章を単語に分割して返す

    >>> tokenize('Bob dropped the apple. Where is the apple?')
    ['Bob', 'dropped', 'the', 'apple', '.', 'Where', 'is', 'the', 'apple', '?']
    '''
    if sent == "":
        print(sent)
        exit(1)
    return [x.strip() for x in re.split('(\W+)+', sent) if x.strip()]


def parse_stories(lines, only_supporting=False):
    '''bAbiタスク形式で与えられたストーリーを解析する'''
    data = []
    story = []
    for line in lines:
        line = line.strip()
        nid, line = line.split(' ', 1)
        nid = int(nid)

        # IDが1になったら新しいストーリーが始まる
        if nid == 1:
            story = []
        # タブが含まれる行は質問とその回答を含む
        if '\t' in line:
            q, a, supporting = line.split('\t')

            # 質問を単語に分割する
            q = tokenize(q)

            # 根拠となるサブストーリーを取得
            # only_supportingがTrueのときはタスクで指定された番号の文章のみ
            # Falseのときはそのストーリーの全文章
            substory = None
            if only_supporting:
                # supportingは複数の文番号が含まれるケースがある
                # ストーリーは1から始まるので1をひく
                supporting = map(int, supporting.split())
                substory = [story[i - 1] for i in supporting]
            else:
                substory = [x for x in story if x]
            data.append((substory, q, a))
            story.append('')
        else:
            sent = tokenize(line)
            story.append(sent)
    return data


def get_stories(filename, only_supporting=False, max_length=None):
    fp = codecs.open(filename, 'r', 'utf-8')

    # dataは質問の数だけタプルがあるリスト
    # data = [ ([supportする文章リスト], 質問文, 回答),  # 1つめの質問
    #          ([supportする文章リスト], 質問文, 回答),  # 2つめの質問
    #          ...
    # ]
    data = parse_stories(fp.readlines(), only_supporting=only_supporting)



if __name__ == '__main__':
    # 対象とするタスク
    # {}にはtrainかtestが入る
    challenge = "tasks_1-20_v1-2/en-10k/qa1_single-supporting-fact_{}.txt"

    print(challenge.format('train'))

    # 訓練データとテストデータを取得
    train_stories = get_stories(challenge.format('train'))
    # test_stories = get_stories(challenge.format('test'))
