VGG16のFine-tuning
2016/12/27

- Keras Blogを参考にしたがこの記事は少し古くてVGG16がKerasの機能として使える前に書かれている
- 今回はKerasに統合されたVGG16を使って書き直す
- CIFAR-10のようにPythonデータの形式でロードできず画像ファイルが与えられる場合
- 画像拡張しない場合でもImageDataGeneratorを使って読み込むと簡単

# Dogs vs. Cats

- Kaggleで公開されているデータセット
- データが欲しい場合はKaggleに登録する必要あり
- train.zipは犬の画像12500枚、猫の画像12500枚から成るデータセット
- 画像を与えて犬か猫か分類する二値分類のタスク
- 犬1000枚・猫1000枚を訓練データ、犬400枚・猫400枚をバリデーションデータとする
- test.zipはKaggleの評価に使うデータらしく正解が付与されていないので今回は使わない

# セットアップ

- Kaggleのサイトからtrain.zipとtest.zipをダウンロード
- train.zipを解凍するとtrainディレクトリができる
- setup.pyを実行すると訓練データ1000枚、バリデーションデータ400枚を以下のように振り分ける
- 分類クラスごとにサブディレクトリを作るのが重要
- KerasのImageDataGeneratorが自動的にcats/dogsをクラス名と認識する

```
data
├── train
│   ├── cats   cat0001.jpg - cat1000.jpg
│   └── dogs   dog0001.jpg - dog1000.jpg
└── validation
    ├── cats   cat0001.jpg - cat0400.jpg
    └── dogs   dog0001.jpg - dog0400.jpg
```

# 小規模な畳み込みニューラルネットをスクラッチから学習

ベースライン
smallcnn

# 学習済みVGG16を特徴抽出器として用いる

vgg16

# 学習済みVGG16の上位層をFine-tuning

fine-tuning

# 参考

- [Building powerful image classification models using very little data](https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html)
- [Dogs vs. Cats](https://www.kaggle.com/c/dogs-vs-cats/)
- [Kerasで学ぶ転移学習](https://elix-tech.github.io/ja/2016/06/22/transfer-learning-ja.html)
