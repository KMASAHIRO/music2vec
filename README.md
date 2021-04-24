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
[SoundNetの論文](https://arxiv.org/pdf/1610.09001.pdf)3ページ目の"2 Large Unlabeled Video Dataset"に従い、入力音源はサンプリングレート22050Hzにし、-256から256までの範囲の数値にした。

## モデルの構成
<img width="500" alt="article" src="https://user-images.githubusercontent.com/74399610/115914638-2ca15480-a4ad-11eb-9c66-c4eccfb0dfd5.png">

[Rajat Hebbarの記事](https://medium.com/@rajatheb/music2vec-generating-vector-embedding-for-genre-classification-task-411187a20820)で紹介されている上の写真のモデルをもとに、下のモデルを作成した。

<img width="800" alt="model_architecture" src="https://user-images.githubusercontent.com/74399610/115947660-1d012a80-a504-11eb-8ed7-7c4d21b1adb4.png">

SoundNetベースのCNNを実際のSoundNetに変えたのが主な変更点。  
それによりCNNの層が増えて入力を長くする必要が出てきたため、入力サイズが大きくなった。

## 結果と評価
[GTZAN](http://opihi.cs.uvic.ca/sound/genres.tar.gz)に含まれる1000個の音楽データの内950個を訓練データとして学習させ、10epochs後と20epochs後に50個のテストデータで推論させて混同行列の作成とAccuracy・Precision・Recall・F1の求値を行った。

<img width="1000" alt="confusion_matrix" src="https://user-images.githubusercontent.com/74399610/115959992-fe258700-a549-11eb-9407-1bb266025b9b.png">

ここで、Precision・Recall・F1の値は各ラベルごとの値をデータ数に応じて重み付けをして平均したものである。例えば、下のような混同行列であれば下の計算式によりPrecisionはおよそ0.58になる。

<img width="500" alt="example_matrix" src="https://user-images.githubusercontent.com/74399610/115948752-001c2580-a50b-11eb-9920-dc2e00be8c48.png">

<img width="300" alt="example_matrix_precision" src=
"https://render.githubusercontent.com/render/math?math=%5Cdisplaystyle+%5Cbegin%7Balign%2A%7D%0A%5Cfrac%7B10%7D%7B19%7D%5Ctimes%5Cfrac%7B15%7D%7B50%7D%2B%5Cfrac%7B12%7D%7B18%7D%5Ctimes%5Cfrac%7B25%7D%7B50%7D%2B%5Cfrac%7B6%7D%7B13%7D%5Ctimes%5Cfrac%7B10%7D%7B50%7D%5Cfallingdotseq0.58%0A%5Cend%7Balign%2A%7D%0A" 
alt="\begin{align*}
\frac{10}{19}\times\frac{15}{50}+\frac{12}{18}\times\frac{25}{50}+\frac{6}{13}\times\frac{10}{50}\fallingdotseq0.58
\end{align*}
">

また、学習前と学習後でモデル中のSoundNetの部分の出力を平坦化したものをPCAやt-SNEで可視化した。
学習後ではt-SNEでジャンルごとにある程度分離されていることがわかる。特にclassicalやmetalはよく分離されている。

<img width="1000" alt="analysis" src="https://user-images.githubusercontent.com/74399610/115950670-abcb7280-a517-11eb-985c-2f72063b5d45.png">
