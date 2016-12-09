import itertools
import numpy as np
from keras.layers import Embedding, SimpleRNN
from keras.models import Sequential

# Simple word embedding
# http://benjaminbolte.com/blog/2016/keras-language-modeling.html

sentences = '''
sam is red
hannah not red
hannah is green
bob is green
bob not red
sam not green
sarah is red
sarah not green'''.strip().split('\n')

is_green = np.asarray([[0, 1, 1, 1, 1, 0, 0, 0]], dtype='int32').T


# 文章を単語のリストに分解
def lemma(sentence):
    return sentence.strip().lower().split(' ')

sentences_lemmatized = [lemma(sentence) for sentence in sentences]

# 文書集合で使われている単語集合
words = set(itertools.chain(*sentences_lemmatized))

word2idx = dict((v, i) for i, v in enumerate(words))
idx2word = list(words)


# 文章を単語インデックスの配列に変換
def to_idx(sentence):
    return [word2idx[word] for word in sentence]


sentences_idx = [to_idx(sentence) for sentence in sentences_lemmatized]
sentences_array = np.asarray(sentences_idx, dtype='int32')

# モデルパラメータ
sentence_maxlen = 3
n_words = len(words)
n_embed_dims = 3

print(sentences_array.shape)
print(is_green.shape)

# Word Embeddingした文章を入力としてred/greenを判別するRNNを構築
model = Sequential()
model.add(Embedding(n_words, n_embed_dims, input_length=sentence_maxlen))
model.add(SimpleRNN(1))

model.summary()
exit()

model.compile(optimizer='sgd', loss='binary_crossentropy')
model.fit(sentences_array, is_green, nb_epoch=5000, verbose=1)
embeddings = model.layers[1].W.get_value()

for i in range(n_words):
    print("%d: %d" % (idx2word[i], embeddings[i]))
