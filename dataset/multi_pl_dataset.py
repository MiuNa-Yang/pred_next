from typing import Optional

import pandas as pd
import pytorch_lightning as pl
import torch
from pytorch_lightning.trainer.supporters import CombinedLoader
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from model.tokenizer import Tokenizer
from utils.data import split_data
from utils.label import Label




class BaseDataset(Dataset):
    def __init__(self, df):
        self.data = self.load_data(df)

    @staticmethod
    def load_data(df):
        return []

    def __getitem__(self, item):
        return self.data[item]

    def __len__(self):
        return len(self.data)


class MutliPLDataset(BaseDataset):
    @staticmethod
    def load_data(df):
        df = df.fillna('')
        df = df[['product_line', 'input_idx', 'output_idx', 'sen_1']]
        # print(df)
        # df.rename(columns={'input_idx': 'input_intent', 'output_idx': 'output_intent', 'sen_1': 'input_sen'}, inplace=True)
        df.rename(columns={'sen_1': 'input_sen'}, inplace=True)

        data = df.to_dict('records')
        return data


label_path = '../data/resources/label.csv'
label = Label(label_path)


def int2idx(x):
    x = tuple(x)
    x = label.int2idx.get(x, 0)
    return x


def int_en_flag2idx(x):
    x = tuple(x)
    return label.int_en_flag2idx.get(x, 0)


DataModuleConf = None


class MultitaskClsDataModule(pl.LightningDataModule):
    def __init__(self, conf):
        super().__init__()
        self.conf = conf
        self.tokenizer = Tokenizer(conf.model_path)
        self.label = Label(conf.label_path)
        self.data = {}
        self.train_data = None
        self.val_data = None
        self.task_num = 0
        self.total_data = pd.read_csv('../data/processed_data/train_data_multi_pl.csv')

    def setup(self, stage: Optional[str] = None):
        pls = set(self.total_data['product_line'])
        self.total_data['input_idx'] = self.total_data[['intent_1', 'product_line']].apply(int_en_flag2idx, axis=1)
        self.total_data['output_idx'] = self.total_data[['intent_2', 'product_line']].apply(int_en_flag2idx, axis=1)
        multi_data = {pl: MutliPLDataset(self.total_data[self.total_data['product_line'] == pl]) for pl in pls}

        multi_data = {pl: split_data(_, self.conf.test_size) for pl, _ in multi_data.items()}
        self.task_num = len(multi_data)
        self.train_data = {k: v[0] for k, v in multi_data.items()}
        self.val_data = {k: v[1] for k, v in multi_data.items()}

    def collate_fn(self, batch):
        # print(batch)
        sent = [_['input_sen'] for _ in batch]
        input_idx = [_['input_idx'] for _ in batch]
        output_idx = [_['output_idx'] for _ in batch]
        sent = self.tokenizer(sent)
        i_lidx = [self.label.idx2lidx[_] for _ in input_idx]
        o_lidx = [self.label.idx2lidx[_] for _ in output_idx]
        input_idx = torch.LongTensor(i_lidx)
        output_idx = torch.LongTensor(o_lidx)
        return {'sent': sent, 'input_idx': input_idx, 'output_idx': output_idx}

    def get_val_dataloader(self, data):
        return DataLoader(data, batch_size=self.conf.val_batch_size,
                          shuffle=False, num_workers=self.conf.num_workers,
                          collate_fn=self.collate_fn, drop_last=False)

    def train_dataloader(self):
        return CombinedLoader({line_id: DataLoader(data, batch_size=self.conf.train_batch_size // self.task_num,
                                                   shuffle=True, num_workers=self.conf.num_workers,
                                                   collate_fn=self.collate_fn, drop_last=True)
                               for line_id, data in self.train_data.items()},
                              mode="max_size_cycle")

    def val_dataloader(self):
        # return CombinedLoader({line_id: self.get_val_dataloader(data)
        #                        for line_id, data in self.val_data.items()}, mode="max_size_cycle")
        val_data = sorted(self.val_data.items())
        return [self.get_val_dataloader(data) for line_id, data in val_data]


class DataTestConfig:
    label_path = '../data/resources/label.csv'
    model_path = '../pretrained_model/rbt3'
    num_workers = 1
    train_batch_size = 18
    val_batch_size = 18
    test_size = 0.1


if __name__ == "__main__":

    config = DataTestConfig
    dm = MultitaskClsDataModule(config)
    dm.setup()
    d = dm.train_dataloader()
    sample = next(iter(d))
    print(sample)
    for i, item in enumerate(d):
        print(item)

