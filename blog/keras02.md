Kerasのインストール
2016/10/20

こういうインストール関係の記事はすぐ時代遅れになるので現在の自分の環境など簡単にまとめておきたい。

- Ubuntu 14.10
- Python 3.5
- Anaconda 4.2.0
- TensorFlow
- Keras 1.1.0

# 1. Ubuntuの仮想マシンを作成

普段はWindowsマシンを使っているが、KerasはWindowsとの相性がとことん悪いなと感じる。

KerasのバックエンドとしてTheanoまたはTensorFlowが選べるのだが、TheanoをGPU有効でWindowsにインストールするのは以外に大変。入った！と思っても実行時に意味不明の長大なエラーが出たりする。TensorFlowはそもそもWindowsに対応していない・・・そんなわけでWindowsに入れるのはもはやあきらめた。

仮想マシンだとホストにGPUがあっても使えないのがちょっと問題。いずれAWSやさくらのGPUインスタンスも試してみたい。

ちなみにAtomやPyCharmなどのGUIエディタを使いたいがため、UbuntuはGUI込みでインストールした。3Dアクセラレータ有効にしてもちょっともっさりする。AtomのRemote-FTPを使って遠隔編集でもよいかも。

# 2. Anaconda Python 3.5 versionをインストール

最近はPythonの公式からインストールすることはほとんどなくなった。NumPy, SciPy, matplotlib, scikit-learn, pandasなど全部入りのAnacondaが最強！あとPython3に移行した。

# 3. TensorFlowのインストール

KerasのデフォルトのバックエンドがTheanoからTensorFlowに切り替わったこともあり、最近ではTensorFlowを使うようにしている。Kerasの前にインストールしておかないとダメ。TensorFlowのインストールページでは、condaで仮想環境を作ろうとなっているが、環境の切り替えが面倒なのでデフォルト環境にそのままインストールしてしまった。

# 4. Kerasのインストール

Kerasはpipで入るのですごく簡単。バックエンドの切り替えは、`~/.keras/keras.json`というファイルに書く。設定ファイルがjsonなのがナウいと思った。

```json
{
    "image_dim_ordering": "tf",
    "epsilon": 1e-07,
    "floatx": "float32",
    "backend": "tensorflow"
}
```

デフォルトの`backend`は`tensorflow`になっている。この設定にある`image_dim_ordering`も実は要注意で`th`か`tf`かによって画像集合を表す4次元テンソルの順番が変わる。これを知らなくてはまったことがある。なぜここだけ`th`とか`tf`とか略称なのだろう・・・という疑問はあるが、デフォルトの設定のままでよいと思う。

# 参考

- [Introduction to Python Deep Learning with Keras](http://machinelearningmastery.com/introduction-python-deep-learning-library-keras/)
