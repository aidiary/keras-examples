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

# 三種類の方法を比較

- スクラッチから
- VGG16を特徴抽出器として利用
- Fine-tuning

【図】

# 小規模な畳み込みニューラルネットをスクラッチから学習

smallcnn.py

- ベースラインとして小規模な畳み込みニューラルネットでスクラッチから学習
- 訓練データが1000枚なのであまり大規模なモデルは学習できない
- 畳み込み層が3つ続くLeNet相当のモデルを構成
- 犬か猫かの分類なので出力層のユニット数は1で活性化関数は0から1の範囲で出力する`sigmoid`を使う。損失関数は`binary_crossentropy`。

- データ拡張のため`ImageDataGenerator`を使う
- 画像の画素値（0-255）を0.0-1.0に正規化する`rescale`を使用
- その他の拡張の詳細は[Kerasによるデータ拡張](http://aidiary.hatenablog.com/entry/20161212/1481549365)（2016/12/12）を参照
- ジェネレータの作り方には、画像集合を表す4Dテンソルを与える`flow()`の他に画像ファイルのディレクトリを与える`flow_from_directory()`が提供されている。
- CIFAR-10のように初めから4Dテンソルがロードできる場合は`flow()`がよいが、今回のように画像ファイルしか与えられていない場合は自分で4Dテンソルに変換するのは面倒なので`flow_from_directory()`を使ったほうが楽
- `target_size`を指定するとリサイズしてくれる
- ディレクトリ構造を把握して勝手にクラス数を認識してくれる

```
Found 2000 images belonging to 2 classes.
```

- 犬クラスと猫クラスのどっちが0でどっちが1に割り当てられたかは`class_indices`でわかる。今回は猫が0で犬が1なのでNNの出力は**犬である確率**（出力が1に近いほど犬で0に近いほど猫）と解釈できる。一般には0.5を閾値として0.5未満なら猫、0.5以上なら犬と判定すればよい。
- どのディレクトリがどのクラスに割り当てられるかは表示してみないとわからないみたい。アルファベット順ではなさそうだ。

```python
> print(train_generator.class_indices)
{'dogs': 1, 'cats': 0}
```

# 学習済みVGG16を特徴抽出器として用いる

- VGG16のFC層を除いた部分を利用する
- VGG16を画像を入力すると `(4, 4, 512)`（4ピクセルx4ピクセルx512チャンネル=8192次元）の特徴ベクトルを出力する特徴抽出器として用いる
- ImageNetには犬や猫の画像も大量に含まれているため犬や猫の特徴をとらえたよい特徴量が抽出できるのではないか？と考える
- 今回のDogs vs. Catsの画像はImageNetではないことに注意。新しい画像の特徴抽出にImageNetで学習したVGG16を使うということ

- 予め訓練データとバリデーションデータの特徴量を抽出してファイルに保存しておく`save_bottleneck_features()`と特徴量を使って多層パーセプトロン（VGG16のFC層と同じ）を学習する`train_top_model()`の二つから成る

- ジェネレータの`shuffle=False`としておく、ディレクトリ内のアルファベット順に生成される？猫が先で犬が後
- `shuffle=True`にしてしまうと犬と猫がバラバラに生成されるので`train_top_model()`で使うラベルがわからなくなる

- 分類は別に多層パーセプトロンを使う必要はなく、SVMなどでもよい

【SVMを使った場合】

# 学習済みVGG16の上位層をFine-tuning

fine-tuning

# 結果と考察


# 参考

- [Building powerful image classification models using very little data](https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html)
- [Dogs vs. Cats](https://www.kaggle.com/c/dogs-vs-cats/)
- [Kerasで学ぶ転移学習](https://elix-tech.github.io/ja/2016/06/22/transfer-learning-ja.html)
