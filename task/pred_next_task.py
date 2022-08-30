from dataclasses import dataclass
from typing import Optional

import torch
from pytorch_lightning import LightningModule
from torch.utils.data import DataLoader

from config.base_config import TrainConfig, DataModuleConfig
from utils.load_config import json_config
from dataset.pred_next_dataset import PredNextDataset
from model.tokenizer import Tokenizer
from model.base_model import ClsModel
from model.focal_loss import FocalLoss
from model.utils import freeze
from utils import schedulers, metrics
from utils.data import random_split_by_ratio
from torch.nn import functional as F


@json_config
@dataclass
class ClsTaskConfig(TrainConfig, DataModuleConfig):
    name: str = ''
    freeze: bool = False


class ClsTask(LightningModule):
    def __init__(self, conf: ClsTaskConfig, from_pretrained=False):
        super().__init__()
        self.save_hyperparameters(ignore='from_pretrained')
        self.conf = conf

        # model
        self.model = ClsModel(model_name=conf.model_name, num_labels=conf.class_num, from_pretrained=from_pretrained)
        if conf.freeze:
            freeze(self.model.backbone)

        self.loss = FocalLoss()

        # data
        self.tokenizer = Tokenizer(conf.tokenizer_name)
        self.train_data = None
        self.val_data = None

    def forward(self, x, label):
        # print(x)
        outputs = self.model(**x)
        loss = self.loss(outputs, label)
        return outputs, loss

    def training_step(self, batch, batch_idx):
        sent, label = batch['sent'], batch['label']
        outputs, loss = self(sent, label)
        self.log('train/loss', loss)
        self.log('lr', self.optimizers().optimizer.state_dict()['param_groups'][0]['lr'])
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        sent, label = batch['sent'], batch['label']
        outputs, loss = self(sent, label)
        self.log('val/loss', loss)
        preds = torch.argsort(outputs, dim=1, descending=True)

        metric_dict = {}
        for k in [1, 5]:
            metric_dict.update(metrics.top_k_accuracy(preds, label, k=k))
        self.log_dict(metric_dict)
        return loss

    def predict_step(self, batch: dict, batch_idx: int, dataloader_idx: int = 0):
        sent = batch['sent']
        outputs = self.model(**sent)
        outputs = F.softmax(outputs, dim=1)
        scores, preds = torch.max(outputs, dim=1)
        return preds.tolist(), scores.tolist()

    def on_predict_epoch_end(self, results):
        preds, scores = [], []
        for _p, _s in results[0]:
            preds += _p
            scores += _s
        results[0] = preds, scores

    def configure_optimizers(self):
        optimizer, scheduler = schedulers.get_linear_schedule_with_warmup(self.model, self.conf, self.trainer)
        return [optimizer], [scheduler]

    def setup(self, stage: Optional[str] = None):
        if stage in ['train', 'validate']:
            if self.conf.val_data_path:
                self.train_data = PredNextDataset(self.conf.train_data_path)
                self.val_data = PredNextDataset(self.conf.val_data_path)
            else:
                data = PredNextDataset(self.conf.data_path)
                train_data, val_data = random_split_by_ratio(data, self.conf.test_size)
                self.train_data = train_data
                self.val_data = val_data
            print(f'train data size: {len(self.train_data)}')
            print(f'val data size: {len(self.val_data)}')

    def collate_fn(self, batch):
        sent, label = zip(*batch)
        sent = list(sent)
        sent = self.tokenizer(sent)
        label = torch.LongTensor(label)
        return {'sent': sent, 'label': label}

    def collate_fn_predict(self, batch):
        sent = [_[0] for _ in batch]
        sent = self.tokenizer(sent)
        return {'sent': sent}

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.conf.train_batch_size,
                          shuffle=True, num_workers=self.conf.num_workers,
                          collate_fn=self.collate_fn, pin_memory=True, persistent_workers=True,
                          drop_last=True)

    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=self.conf.val_batch_size,
                          shuffle=False, num_workers=self.conf.num_workers,
                          collate_fn=self.collate_fn, pin_memory=True, persistent_workers=True,
                          drop_last=False)

    def get_predict_dataloader(self, data, batch_size):
        return DataLoader(data, batch_size=batch_size,
                          shuffle=False, num_workers=self.conf.num_workers,
                          collate_fn=self.collate_fn_predict, pin_memory=True, persistent_workers=True,
                          drop_last=False)

    def check_dataloader(self):
        return DataLoader(self.val_data, batch_size=2, collate_fn=self.collate_fn)
