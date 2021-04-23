import tensorflow as tf
import numpy as np
import librosa
from . import data_preparation

from tensorflow.keras.layers import BatchNormalization, Activation, Conv1D, MaxPooling1D, ZeroPadding1D, InputLayer
from tensorflow.keras.models import Sequential

# 事前学習済みの重みを用いてSoundNetを構築する関数(https://github.com/pseeth/soundnet_keras/blob/master/soundnet.py より借用)
def build_model():
    """
    Builds up the SoundNet model and loads the weights from a given model file
    :return model: SoundNetのモデル
    """

    model_weights = np.load('./sound8.npy', allow_pickle=True, encoding='bytes').item()

    keys = list()
    for key in model_weights.keys():
        keys.append(key)
    for name in keys:
        model_weights[name.decode('utf-8')] = model_weights[name]
        model_weights.pop(name)
        ch_keys = list()
        for key in model_weights[name.decode('utf-8')]:
            ch_keys.append(key)
        for ch_name in ch_keys:
            model_weights[name.decode('utf-8')][ch_name.decode('utf-8')] = model_weights[name.decode('utf-8')][ch_name]
            model_weights[name.decode('utf-8')].pop(ch_name)

    model = Sequential()
    model.add(InputLayer(batch_input_shape=(None, None, 1)))

    filter_parameters = [{'name': 'conv1', 'num_filters': 16, 'padding': 32,
                          'kernel_size': 64, 'conv_strides': 2,
                          'pool_size': 8, 'pool_strides': 8},

                         {'name': 'conv2', 'num_filters': 32, 'padding': 16,
                          'kernel_size': 32, 'conv_strides': 2,
                          'pool_size': 8, 'pool_strides': 8},

                         {'name': 'conv3', 'num_filters': 64, 'padding': 8,
                          'kernel_size': 16, 'conv_strides': 2},

                         {'name': 'conv4', 'num_filters': 128, 'padding': 4,
                          'kernel_size': 8, 'conv_strides': 2},

                         {'name': 'conv5', 'num_filters': 256, 'padding': 2,
                          'kernel_size': 4, 'conv_strides': 2,
                          'pool_size': 4, 'pool_strides': 4},

                         {'name': 'conv6', 'num_filters': 512, 'padding': 2,
                          'kernel_size': 4, 'conv_strides': 2},

                         {'name': 'conv7', 'num_filters': 1024, 'padding': 2,
                          'kernel_size': 4, 'conv_strides': 2},

                         {'name': 'conv8_2', 'num_filters': 401, 'padding': 0,
                          'kernel_size': 8, 'conv_strides': 2},
                         ]

    for x in filter_parameters:
        model.add(ZeroPadding1D(padding=x['padding']))
        model.add(Conv1D(x['num_filters'],
                         kernel_size=x['kernel_size'],
                         strides=x['conv_strides'],
                         padding='valid'))
        weights = model_weights[x['name']]['weights'].reshape(model.layers[-1].get_weights()[0].shape)
        biases = model_weights[x['name']]['biases']

        model.layers[-1].set_weights([weights, biases])

        if 'conv8' not in x['name']:
            gamma = model_weights[x['name']]['gamma']
            beta = model_weights[x['name']]['beta']
            mean = model_weights[x['name']]['mean']
            var = model_weights[x['name']]['var']

            model.add(BatchNormalization())
            model.layers[-1].set_weights([gamma, beta, mean, var])
            model.add(Activation('relu'))
        if 'pool_size' in x:
            model.add(MaxPooling1D(pool_size=x['pool_size'],
                                   strides=x['pool_strides'],
                                   padding='valid'))

    return model

# music2vecを用いた音楽ジャンル分類モデルを構築する関数
def create_model():
    '''
    :return model: music2vecを用いた音楽ジャンル分類モデル
    '''
    soundnet = build_model()

    music2vec_input = tf.keras.Input(shape=(675808,1))
    x = soundnet(music2vec_input)
    x = tf.keras.layers.LSTM(200,return_sequences=True)(x)
    x = tf.keras.layers.LSTM(200,return_sequences=True)(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(400)(x)
    x = tf.keras.layers.Dense(100)(x)
    music2vec_output = tf.keras.layers.Dense(10,activation='softmax')(x)
    music2vec = tf.keras.Model(inputs=music2vec_input,outputs=music2vec_output)

    return music2vec

# gtzanのデータで学習させる関数
def train(model, train_filenames, train_labels, batch_size, epochs, initial_epoch=0, validation_data=None):
    '''
    :param model: music2vecを用いた音楽ジャンル分類モデル
    :param train_filenames: 入力データとなる音楽ファイルパス(listまたはnp.ndarray)
    :param train_labels: 入力データのラベル(listまたはnp.ndarray)
    :param batch_size: バッチサイズ(int)
    :param epochs: 関数内で行う学習epoch数(int)
    :param initial_epoch: 関数実行前の学習epochs数。デフォルトでは0。(int)
    :param validation_data: バリデーションデータ(任意)。(モデルへの入力,正解ラベル)の形のタプル。(tuple)
    :return model: 学習後のモデル
    '''

    model.compile(optimizer='rmsprop',loss='CategoricalCrossentropy',metrics=['accuracy'])
    steps = len(train_filenames)//batch_size

    if validation_data is None:
        model.fit(x=data_preparation.generator(train_filenames, train_labels, batch_size),
                      steps_per_epoch=steps, epochs=epochs, initial_epoch=initial_epoch)
    else:
        model.fit(x=data_preparation.generator(train_filenames, train_labels, batch_size),
                      steps_per_epoch=steps, validation_data=validation_data,
                      epochs=epochs, initial_epoch=initial_epoch)

    return model