from typing import Optional
import pickle
import pandas as pd
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
from model.tokenizer import Tokenizer
from config.base_config import DataModuleConfig
from dataset.pred_next_dataset import PredNextDataset


class PredNextDataModule(pl.LightningDataModule):
    def __init__(self, cfg: DataModuleConfig):
        super().__init__()
        self.cfg = cfg
        self.tokenizer = Tokenizer(cfg.tokenizer_name, model_max_length=cfg.max_len)
        self.data = {}
        self.train_data = None
        self.test_data = None
        self.intent2index = None

    def setup(self, stage: Optional[str] = None):
        train_data = PredNextDataset(self.cfg.train_data_path)
        test_data = PredNextDataset(self.cfg.test_data_path)

        with open(self.cfg.intent2index_path, 'rb') as f:
            self.intent2index = pickle.load(f)

        self.train_data = train_data
        self.test_data = test_data
        print(f'train data size: {len(self.train_data)}')
        print(f'val data size: {len(self.test_data)}')

    def collate_fn(self, batch):
        input_intent, input_sent, output_intent = zip(*batch)
        input_sent = list(input_sent)
        input_sent = self.tokenizer(input_sent)

        output_intent = [self.intent2index[_] for _ in output_intent]
        input_intent = [self.intent2index[_] for _ in input_intent]

        output_intent = torch.LongTensor(output_intent)
        input_intent = torch.LongTensor(input_intent)
        return {'input_intent': input_intent, 'input_sent': input_sent, 'output_intent': output_intent}

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.cfg.train_batch_size, shuffle=True
                          , collate_fn=self.collate_fn, pin_memory=True, persistent_workers=True,
                          drop_last=True)

    def val_dataloader(self):
        return DataLoader(self.test_data, batch_size=self.cfg.test_batch_size, shuffle=False
                          , collate_fn=self.collate_fn, pin_memory=True, persistent_workers=True,
                          drop_last=False)

    def check_dataloader(self):
        return DataLoader(self.test_data, batch_size=2, collate_fn=self.collate_fn, shuffle=True)


if __name__ == "__main__":

    from config.base_config import DataModuleConfig
    from config import data_config

    cfg_1 = DataModuleConfig(test_data_path=data_config.test_data_path, train_data_path=data_config.train_data_path,
                             intent2index_path=data_config.intent2index_path, tokenizer_name=data_config.roberta_path,
                             max_len=40)

    dm = PredNextDataModule(cfg_1)
    dm.setup()
    dl = dm.check_dataloader()
    sample = next(iter(dl))

    print(sample)
