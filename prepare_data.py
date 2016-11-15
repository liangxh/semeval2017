from keras.utils import np_utils
from keras.preprocessing import sequence

from common import data_manager, input_adapter, wordembed

import numpy as np
# np.random.seed(1337)  # for reproducibility


def prepare_dataset(key_subtask, vocabs):
    text_indexer = input_adapter.get_text_indexer(vocabs)
    label_indexer = input_adapter.get_label_indexer(key_subtask)

    def prepare_dataset(mode):  # 函数里定义函数?
        texts_labels = data_manager.read_texts_labels(key_subtask, mode)
        x, y = input_adapter.adapt_texts_labels(texts_labels, text_indexer, label_indexer)
        return x, y

    dataset = tuple([prepare_dataset(mode) for mode in ['train', 'dev', 'devtest']])

    return dataset

def prepare_input(xy, input_length):
    x, y = xy
    x = sequence.pad_sequences(x, maxlen = input_length)
    y = np_utils.to_categorical(y)
    return x, y