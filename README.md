# music2vec
SoundNetをベースにしたmusic2vecを用いた音楽ジャンルの分類(参考：https://medium.com/@rajatheb/music2vec-generating-vector-embedding-for-genre-classification-task-411187a20820)

## 概要
[Rajat Hebbarの記事](https://medium.com/@rajatheb/music2vec-generating-vector-embedding-for-genre-classification-task-411187a20820)のSoundNet + Stacked-LSTMモデルを参考にしたSoundNetをベースにしたmusic2vecを用いた音楽ジャンル分類モデルの実装。  
[GTZAN](http://opihi.cs.uvic.ca/sound/genres.tar.gz)のデータを用いた音楽ジャンル分類を学習させた後、テストデータでの推論結果を混同行列にまとめ、Accuracy・Precision・Recall・F1の値を求めた(Precision・Recall・F1はデータ数で重みを付けて平均をとった)。  
また、学習前と学習後でモデル中のSoundNetの部分の出力を平坦化したものをPCAやt-SNEで可視化した。  
music2vecディレクトリにはモデルの実装、学習を簡単にできるようにモジュールとしてまとめており、パッケージとして利用することが可能。  
[Google Colaboratory](https://colab.research.google.com/drive/1TlhN6ZW9ytXwIsxFFB0fYdUSfuz5DwLz?usp=sharing)上にこれらを実行したものをまとめている(music2vec.ipynbと同じもの)。
