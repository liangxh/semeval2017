#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: xiwen zhao
@created: 2016.12.3
"""


from optparse import OptionParser
from trainer import BaseTrainer

from keras.models import model_from_json
from keras.optimizers import RMSprop, SGD


class Trainer(BaseTrainer):
    def get_model_name(self):
        return __file__.split('/')[-1].split('.')[0]

    def set_merge_num(self, merge_num):
        self.merge_num = merge_num

    def post_prepare_X(self, x):
        return [x for i in range(self.merge_num)]

    def set_model_config(self, options):
        self.config = dict(
            optimizer = options.optimizer,
            model_name = options.model_name,
        )

    def get_optimizer(self, key_optimizer):
        if key_optimizer == 'rmsprop':
            return RMSprop(lr=0.001, rho=0.9, epsilon=1e-06)
        else:  # 'sgd'
            return SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=False)

    def build_model(self, config, weights):
        fname = '../data/model/subtask%s_%s_config_new.json' % (self.key_subtask, config['model_name'])
        json_string = open(fname, 'r').read()
        model = model_from_json(json_string)

        if config['nb_classes'] > 2:
            loss_type = 'categorical_crossentropy'
        else:
            loss_type = 'binary_crossentropy'

        model.compile(loss=loss_type,
                      optimizer=self.get_optimizer(config['optimizer']),
                      metrics=['accuracy'])

        return model


def main():
    optparser = OptionParser()
    optparser.add_option("-t", "--task", dest="key_subtask", default="D")
    optparser.add_option("-p", "--nb_epoch", dest="nb_epoch", type="int", default=50)
    optparser.add_option("-e", "--embedding", dest="fname_Wemb", default="glove.twitter.27B.25d.txt.trim")
    optparser.add_option("-o", "--optimizer", dest="optimizer", default="rmsprop")
    optparser.add_option("-m", "--model_name", dest="model_name", default="finki")
    optparser.add_option("-n", "--merge_num", dest="merge_num", type="int", default=2)
    opts, args = optparser.parse_args()

    trainer = Trainer(opts)
    trainer.set_merge_num(opts.merge_num)
    trainer.pred_prob()

if __name__ == '__main__':
    main()
