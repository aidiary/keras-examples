import numpy as np
np.random.seed(1337)  # for reproducibility

from keras.datasets import reuters
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.callbacks import EarlyStopping
from keras.utils import np_utils
from keras.preprocessing.text import Tokenizer

if __name__ == '__main__':
    max_words = 1000
    batch_size = 32
    nb_epoch = 100

    # Reutersデータセットをロード
    # Xは各文章に含まれる単語インデックスのリストで表される
    # 単語インデックスがmax_wordsまでの単語のみ残す
    # TODO: the, ofなどのstop wordsは除去したほうがよい？
    (X_train, y_train), (X_test, y_test) = reuters.load_data(nb_words=max_words, test_split=0.2)

    print(len(X_train), 'train sequences')
    print(len(X_test), 'test sequences')

    # # word => indexの辞書を取得
    # word_index = reuters.get_word_index()
    #
    # # index => wordの辞書を作成
    # index_word = dict([(v, k) for k, v in word_index.items()])
    #
    # for i in range(1, max_words + 1):
    #     print(i, index_word[i])

    # 文書のクラス数を計算
    nb_classes = np.max(y_train) + 1
    print(nb_classes, 'classes')

    # 行が文書、列が単語になる行列に変換
    # 0:文書がその単語を含まない
    # 1:文書がその単語を含む
    print('Vectorizing sequence data...')
    tokenizer = Tokenizer(nb_words=max_words)
    X_train = tokenizer.sequences_to_matrix(X_train, mode='binary')
    X_test = tokenizer.sequences_to_matrix(X_test, mode='binary')
    print('X_train shape:', X_train.shape)
    print('X_test shape:', X_test.shape)

    # クラスラベルをone-hot-encoding
    print('Convert class vector to binary class matrix (for use with categorical_crossentropy)')
    Y_train = np_utils.to_categorical(y_train, nb_classes)
    Y_test = np_utils.to_categorical(y_test, nb_classes)
    print('Y_train shape:', Y_train.shape)
    print('Y_test shape:', Y_test.shape)

    # 多層パーセプトロン
    print('Building model...')
    model = Sequential()
    model.add(Dense(512, input_shape=(max_words,)))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))

    # Early-stopping
    early_stopping = EarlyStopping()

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    history = model.fit(X_train, Y_train,
                        nb_epoch=nb_epoch,
                        batch_size=batch_size,
                        verbose=1,
                        validation_split=0.1,
                        callbacks=[early_stopping])

    loss, acc = model.evaluate(X_test, Y_test,
                               batch_size=batch_size,
                               verbose=0)

    print('Test acc:', acc)
