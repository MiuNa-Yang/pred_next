import os
from os.path import join


ROOT = os.path.abspath(join(os.path.dirname('__file__'), '..'))
DATA_DIR = join(ROOT, 'data')


class MultiPLPredNext:

    # model config
    intent_num_dic = {}

    # data module config
    train_data_path = ''
    test_data_path = ''
    val_data_path = ''
    label_path = ''

    # train config
    test_size = 0.1
    train_batch_size = 16
    test_batch_size = 16
    val_batch_size = 16
    intent2index_path = ''
    index2intent_path = ''
    tokenizer_name = ''
    bert_path = ''
    max_len = 64


