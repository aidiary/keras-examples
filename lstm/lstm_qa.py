"""
Facebook bAbiデータを適用したLSTMエンコーダ
https://www.youtube.com/watch?v=YimQOpSRULY&t=885s
https://bitbucket.org/fchollet/keras_workshop
http://www.thespermwhale.com/jaseweston/babi/tasks_1-20_v1-2.tar.gz
"""

from keras.models import Sequential
from keras.layers.embeddings import Embedding
from keras.layers.core import Activation, Dense, Merge
from keras.layers.recurrent import GRU, LSTM
from keras.preprocessing.sequence import pad_sequences
import re
import codecs
import numpy as np
import itertools


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

    def flatten(lst):
        '''リストをフラット化（1次元配列）する
        http://docs.python.jp/2/library/itertools.html'''
        return list(itertools.chain.from_iterable(lst))

    data = [(flatten(story), q, answer) for story, q, answer in data if not max_length or len(flatten(story)) < max_length]
    return data


def vectorize_stories(data, word_idx, story_maxlen, query_maxlen):
    X = []
    Xq = []
    Y = []
    for story, query, answer in data:
        x = [word_idx[w] for w in story]
        xq = [word_idx[w] for w in query]
        # 正解の単語のインデックスのみ1
        y = np.zeros(len(word_idx) + 1)  # 0は予約
        y[word_idx[answer]] = 1
        X.append(x)
        Xq.append(xq)
        Y.append(y)

    # 時系列データをパディング
    # >>> pad_sequences([[1,2], [1,2,3], [1], [1,2,3,4,5]], 5)
    # array([[0, 0, 0, 1, 2],
    #        [0, 0, 1, 2, 3],
    #        [0, 0, 0, 0, 1],
    #        [1, 2, 3, 4, 5]], dtype=int32)
    return pad_sequences(X, maxlen=story_maxlen), pad_sequences(Xq, maxlen=query_maxlen), np.array(Y)


if __name__ == '__main__':
    # 対象とするタスク
    # {}にはtrainかtestが入る
    challenge = "tasks_1-20_v1-2/en-10k/qa1_single-supporting-fact_{}.txt"

    print(challenge.format('train'))

    # 訓練データとテストデータを取得
    train_stories = get_stories(challenge.format('train'))
    test_stories = get_stories(challenge.format('test'))

    # (根拠文, 質問文, 回答) の組の数
    print("# of train_stories", len(train_stories))
    print("# of test_stories", len(test_stories))

    vocab = []
    for story, q, answer in train_stories + test_stories:
        vocab.extend(story)
        vocab.extend(q)
        vocab.extend([answer])
    vocab = sorted(set(vocab))

    # 0は使わない
    vocab_size = len(vocab) + 1

    # 根拠文と質問の最大単語数を取得
    story_maxlen = max(map(len, (x for x, _, _ in train_stories + test_stories)))
    query_maxlen = max(map(len, (x for _, x, _ in train_stories + test_stories)))

    print(vocab_size)
    print(story_maxlen)
    print(query_maxlen)

    # 単語をインデックスに変換
    word_idx = dict((c, i + 1) for i, c in enumerate(vocab))
    inputs_train, queries_train, answers_train = vectorize_stories(train_stories, word_idx, story_maxlen, query_maxlen)
    inputs_test, queries_test, answers_test = vectorize_stories(test_stories, word_idx, story_maxlen, query_maxlen)

    print(inputs_train[0])
    print(queries_train[0])
    print(answers_train[0])

    # モデルを構築
    input_encoder = Sequential()
    input_encoder.add(Embedding(input_dim=vocab_size, output_dim=64))
    input_encoder.add(LSTM(64, return_sequences=False))

    question_encoder = Sequential()
    question_encoder.add(Embedding(input_dim=vocab_size, output_dim=64))
    question_encoder.add(LSTM(64, return_sequences=False))

    model = Sequential()
    model.add(Merge([input_encoder, question_encoder, ]))
