from sklearn.datasets import load_boston
from sklearn.model_selection import cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor

def deeper_model():
    """隠れ層を2つ持つモデルを構築"""
    model = Sequential()
    model.add(Dense(13, input_dim=13, init='normal', activation='relu'))
    model.add(Dense(6, init='normal', activation='relu'))
    model.add(Dense(1, init='normal'))

    model.compile(loss='mean_squared_error',
                  optimizer='adam')

    return model

if __name__ == "__main__":
    # Boston house priceデータセットをロード
    boston = load_boston()
    X = boston.data
    Y = boston.target

    # データ正規化、回帰を行うパイプラインを作成
    estimators = []
    estimators.append(('standardize',
                       StandardScaler()))
    estimators.append(('regression',
                       KerasRegressor(build_fn=deeper_model,
                                      nb_epoch=50,
                                      batch_size=5,
                                      verbose=0)))
    pipeline = Pipeline(estimators)

    # 10-fold Cross Validationでモデル評価
    kfold = KFold(n_splits=10)
    results = cross_val_score(pipeline, X, Y, cv=kfold)
    print("Deeper MSE: %.2f" % results.mean())
