KerasでMNIST

2016/11/7

- MNISTデータのロード
- モデルの可視化
- Early-stoppingによる収束判定
    train_lossは下がるのにval_lossが上がっていく->過学習！
    過学習する前に学習を打ち切りたい
    patienceを上げると打ち切るのを我慢しやすくなる（0, 5, 10）
    デフォルトでOKか？
    verboseが機能していない？どこに表示される？
- 学習履歴のプロット
