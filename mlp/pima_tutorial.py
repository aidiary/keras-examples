import os
import numpy as np
from keras.models import Sequential
from keras.layers import Dense

# 参考
# http://machinelearningmastery.com/tutorial-first-neural-network-python-keras/

seed = 7
np.random.seed(seed)

if __name__ == "__main__":
    # Pima indians datasetをロード
    dataset = np.loadtxt(os.path.join("..", "data", "pima-indians-diabetes.data"), delimiter=',')

    # データとラベルを取得
    X = dataset[:, 0:8]
    Y = dataset[:, 8]

    # 多層ニューラルネットワークモデルを構築
    model = Sequential()
    model.add(Dense(12, input_dim=8, init='uniform', activation='relu'))
    model.add(Dense(8, init='uniform', activation='relu'))
    model.add(Dense(1, init='uniform', activation='sigmoid'))

    # モデルをコンパイル
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    # モデルを学習
    model.fit(X, Y, nb_epoch=150, batch_size=10)

    # モデルを評価
    scores = model.evaluate(X, Y, verbose=0)
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))

    # 訓練データに対してモデルで出力を予測
    # 0.0-1.0の確率値で出力されるため0.5以上の場合はクラス1、未満の場合は0と判定する
    predictions = np.round(model.predict(X))
    correct = Y[:, np.newaxis]
