import tensorflow as tf
from tensorflow import keras
import numpy as np


def decode_review(text):
    word_index = imdb.get_word_index()

    # 保留第一个索引
    word_index = {k: (v + 3) for k, v in word_index.items()}
    word_index['<PAD>'] = 0
    word_index['<START>'] = 1
    word_index['<UNK>'] = 2
    word_index["<UNUSED>"] = 3

    reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
    return ' '.join([reverse_word_index.get(i, '?') for i in text])


if __name__ == '__main__':
    imdb = keras.datasets.imdb
    (train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)  # 选择训练中最常出现的一万个词
    print("Training entries: {}, labels: {}".format(len(train_data), len(train_labels)))
    print(len(train_data[0]), len(train_data[1]))
    # 一个映射单词到整数索引的词典
    print(decode_review(train_data[0]))