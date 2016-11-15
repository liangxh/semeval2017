from __future__ import print_function
import numpy as np
np.random.seed(1337)  # for reproducibility

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Embedding
from keras.layers import LSTM, SimpleRNN, GRU
from prepare_data import prepare_dataset, prepare_input
from common import data_manager, wordembed

max_features = 20000
input_length = 45
batch_size = 32

key_subtask = 'A'
wemb_file = 'glove.twitter.27B.25d.txt'

print('Loading data...')
vocabs = data_manager.read_vocabs(key_subtask)
dataset = prepare_dataset(key_subtask, vocabs)
train, valid, test = map(lambda dset: prepare_input(dset, input_length), dataset)

weights = dict(
    Wemb = wordembed.get(vocabs, wemb_file),
)

config = dict(
    # parameters related to the dataset
    nb_classes = len(set(dataset[0][1])),  # use set() to filter repetitive classes
    max_features = len(vocabs),
    input_length = input_length,
    embedding_dims = weights['Wemb'].shape[1],

)

print('Build GRNN model...')
model = Sequential()
model.add(Embedding(config['max_features'],
                        config['embedding_dims'],
                        input_length = config['input_length'],
                        weights = [weights['Wemb']] if 'Wemb' in weights else None,
                        dropout = 0.2))
model.add(GRU(128, dropout_W=0.2, dropout_U=0.2))
# model.add(Dense(1))
model.add(Dense(config['nb_classes']))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

print('Train...')
model.fit(train[0], train[1], batch_size=batch_size, nb_epoch=15,
          validation_data=valid)

score, acc = model.evaluate(test[0], test[1],
                            batch_size=batch_size)

print('Test accuracy:', acc)

'''
loss='binary_crossentropy' optimizer = 'adam' 0.73782
'''