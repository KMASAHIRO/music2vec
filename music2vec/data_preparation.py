import numpy as np
import librosa

# gtzanの音楽ファイルパスとラベルを作成し、訓練データとテストデータに分けて返す関数
def gtzan_preparation(genres_path, testdata_num, random_seed=0):
    '''
    :param genres_path: gtzanのデータが格納されているgenresディレクトリのパス(str)
    :param testdata_num: テストデータの数(全データは1000)(int)
    :param random_seed: データをシャッフルするときのseed(int)
    :return train_filenames: 訓練データの音楽ファイルパス(np.ndarray)
    :return test_filenames: テストデータの音楽ファイルパス(np.ndarray)
    :return train_labels: 訓練データのラベル(np.ndarray)
    :return test_labels: テストデータのラベル(np.ndarray)
    '''
    # ロードする音楽ファイル名の作成
    genre_list = ["blues", "classical", "country", "disco", "hiphop", "jazz", "metal", "pop", "reggae", "rock"]

    gtzan_filenames = list()
    for genre in genre_list:
        for i in range(100):
            gtzan_filenames.append(genres_path + genre + "/" + genre + ".000{:02}.wav".format(i))
    gtzan_filenames = np.asarray(gtzan_filenames)

    # ラベルの作成
    gtzan_labels = list()
    for i in range(10):
        for k in range(100):
            gtzan_labels.append(tf.one_hot(i, 10))
    gtzan_labels = np.asarray(gtzan_labels)

    # データをシャッフル(最初はジャンルごとにソートされている)
    np_random = np.random.RandomState(seed=1)
    new_gtzan_filenames = np_random.permutation(gtzan_filenames)
    np_random = np.random.RandomState(seed=1)
    new_gtzan_labels = np_random.permutation(gtzan_labels)

    # 訓練データとテストデータに分割
    train_filenames = new_gtzan_filenames[:-testdata_num]
    test_filenames = new_gtzan_filenames[-testdata_num:]
    train_labels = new_gtzan_labels[:-testdata_num]
    test_labels = new_gtzan_labels[-testdata_num:]

    return train_filenames, test_filenames, train_labels, test_labels


# 音楽ファイルをロードして訓練データを生成するジェネレータ
def generator(filenames, labels, batch_size):
    '''
    :param filenames: 音楽ファイルのパス(listまたはnp.ndarray)
    :param labels: 音楽ジャンルを示すラベル(listまたはnp.ndarray)
    :param batch_size: バッチサイズ(int)
    :yield train_inputs: music2vecに入力するデータ(np.ndarray)
    :yield train_labels: music2vecに入力するデータのラベル(np.ndarray)
    '''
    index = 0
    seed_num = 0
    new_filenames = filenames.copy()
    new_labels = labels.copy()
    while True:
        train_inputs = list()
        train_labels = list()
        # データの最後まで来たらシャッフルしてデータの最初に戻る
        if index + batch_size > len(filenames):
            index = 0
            np_random = np.random.RandomState(seed=seed_num)
            new_filenames = np_random.permutation(new_filenames)
            np_random = np.random.RandomState(seed=seed_num)
            new_labels = np_random.permutation(new_labels)
            seed_num += 1

        # バッチサイズの分だけ音楽をロードしてSoundNetに合うように前処理する
        for i in range(batch_size):
            y, sr = librosa.load(new_filenames[index], sr=22050, mono=True, dtype=np.float32)
            y = np.concatenate((y, np.zeros(675808 - len(y))), axis=0)
            y *= 256.0
            y = np.reshape(y, (-1, 1))
            train_inputs.append(y)
            train_labels.append(new_labels[index])
            index += 1
        train_inputs = np.asarray(train_inputs)
        train_labels = np.asarray(train_labels)

        yield train_inputs, train_labels

# 音楽ファイルをロードしてSoundNetに合うように前処理する関数
def load_audiofile(filename):
    '''
    :param filename:  音楽ファイルパス(str)
    :return y: 音楽データ(np.ndarray)
    '''
    y,sr = librosa.load(filename,sr=22050,mono=True,dtype=np.float32)
    y *= 256.0
    y = np.reshape(y,(-1,1))
    return y