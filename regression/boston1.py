from sklearn.datasets import load_boston
from sklearn.model_selection import cross_val_score, KFold

from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor

def baseline_model():
    """隠れ層を1つ持つ単純なモデルを構築"""
    model = Sequential()
    model.add(Dense(13, input_dim=13, init='normal', activation='relu'))
    model.add(Dense(1, init='normal'))

    model.compile(loss='mean_squared_error',
                  optimizer='adam')

    return model

if __name__ == "__main__":
    # Boston house priceデータセットをロード
    boston = load_boston()
    X = boston.data
    Y = boston.target

    # scikit-learnで使えるようにラッパーをつける
    estimator = KerasRegressor(build_fn=baseline_model,
                               nb_epoch=100,
                               batch_size=5,
                               verbose=0)

    # 10-fold Cross Validationでモデル評価
    kfold = KFold(n_splits=10)
    results = cross_val_score(estimator, X, Y, cv=kfold)
    print("MSE: %.2f" % results.mean())
