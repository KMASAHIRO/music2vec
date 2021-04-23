# music2vec
SoundNetをベースにしたmusic2vecを用いた音楽ジャンルの分類(参考：https://medium.com/@rajatheb/music2vec-generating-vector-embedding-for-genre-classification-task-411187a20820)

## 概要
[Rajat Hebbarの記事](https://medium.com/@rajatheb/music2vec-generating-vector-embedding-for-genre-classification-task-411187a20820)のSoundNet + Stacked-LSTMモデルを参考にしたSoundNetをベースにしたmusic2vecを用いた音楽ジャンル分類モデルの実装。  
[GTZAN](http://opihi.cs.uvic.ca/sound/genres.tar.gz)のデータを用いた音楽ジャンル分類を学習させた後、テストデータでの推論結果を混同行列にまとめ、Accuracy・Precision・Recall・F1の値を求めた(Precision・Recall・F1はデータ数で重みを付けて平均をとった)。  
また、学習前と学習後でモデル中のSoundNetの部分の出力を平坦化したものをPCAやt-SNEで可視化した。  
music2vecディレクトリにはモデルの実装、学習を簡単にできるようにモジュールとしてまとめており、パッケージとして利用することが可能。  
[Google Colaboratory](https://colab.research.google.com/drive/1TlhN6ZW9ytXwIsxFFB0fYdUSfuz5DwLz?usp=sharing)上にこれらを実行したものをまとめている(music2vec.ipynbと同じもの)。

## 実行したPythonのバージョンと使用したパッケージ

- Python 3.7.10
- Tensorflow 2.4.1
- numpy 1.19.5
- librosa 0.8.0

## データの前処理
[論文](https://arxiv.org/pdf/1610.09001.pdf)3ページ目の"2 Large Unlabeled Video Dataset"に従い、入力音源はサンプリングレート22050Hzにし、-256から256までの範囲の数値にした。
