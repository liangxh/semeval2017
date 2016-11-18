#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: xiwen zhao
@created: 2016.11.17
"""

from optparse import OptionParser
from trainer import BaseTrainer
from common import data_manager

from keras.models import Sequential
<<<<<<< HEAD
from keras.layers import Dense, Dropout, Activation, Embedding, Merge
from keras.layers import LSTM, SimpleRNN, GRU
from prepare_data import prepare_dataset, prepare_input
from common import data_manager, wordembed, input_adapter

max_features = 20000
input_length = 45
batch_size = 32

key_subtask = 'C'
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

    nb_filter = 250,
    filter_length = 3,
    hidden_dims = 250,
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

print('Train grnn...')
model.fit(train[0], train[1], batch_size=batch_size, nb_epoch=15, validation_data=valid)

from cnn import main
model_cnn = main()

# merged model
merged_model = Sequential()
merged_model.add(Merge([model, model_cnn], mode='concat', concat_axis=1))

merged_model.add(Dense(config['nb_classes']))
merged_model.add(Activation('softmax'))

merged_model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

print('Train merged_model...')
merged_model.fit([train[0], train[0]], train[1],
                 batch_size=batch_size, nb_epoch=15,
                 validation_data=[[valid[0],valid[0]],valid[1]]
                 )

score, acc = merged_model.evaluate([test[0], test[0]], test[1],
                           batch_size=batch_size)

print('Test accuracy:', acc)

pred_classes = merged_model.predict_classes([test[0], test[0]], batch_size=batch_size, verbose=1)

tweet_id = []
tweet_topic = []
tweet_label = []
tweet_id_topic_label = data_manager.read_id_topic_label(key_subtask) # subtask BC
# tweet_id_label = data_manager.read_id_label(key_subtask)  # subtask A

for item in tweet_id_topic_label:
    tweet_id.append(item[0])
    tweet_topic.append(item[1])
    tweet_label.append(item[2])

f_pred = open('pred_resultC.txt', 'w')

for t_id, t_topic, result in zip(tweet_id, tweet_topic, pred_classes):
    indexer = input_adapter.get_label_indexer(key_subtask)
    f_pred.write(t_id + '\t' + t_topic + '\t' + indexer.label(result) + '\n')
f_pred.close()

f_gold = open('gold_resultC.txt', 'w')

for t_id, t_topic, t_label in zip(tweet_id, tweet_topic, tweet_label):
    f_gold.write(t_id + '\t' + t_topic + '\t' + t_label + '\n')
f_gold.close()
=======
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding
from keras.layers import GRU
from keras.optimizers import SGD

class Trainer(BaseTrainer):
    def set_model_config(self, options):
        self.config = dict(
            nb_filter = options.nb_filter,
            filter_length = options.filter_length,
            hidden_dims = options.hidden_dims,
        )

    def build_model(self, config, weights):
        print 'Build grnn model...'
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

        return model


def main():
    optparser = OptionParser()
    optparser.add_option("-t", "--task", dest = "key_subtask", default = "B")
    optparser.add_option("-e", "--embedding", dest = "fname_Wemb", default = "glove.twitter.27B.25d.txt")
    optparser.add_option("-d", "--hidden_dims", dest = "hidden_dims", type = "int", default = 250)
    optparser.add_option("-f", "--nb_filter", dest = "nb_filter", type = "int", default = 250)
    optparser.add_option("-l", "--filter_length", dest = "filter_length", type = "int", default = 3)
    opts, args = optparser.parse_args()

    trainer = Trainer(opts)
    model = trainer.train()
    
    test = data_manager.read_texts_labels(opts.key_subtask, 'devtest')
    trainer.evaluate(test)


if __name__ == '__main__':
    main()

>>>>>>> 388be2bc55fc0c055325740718ff2f4a10dd2f18
