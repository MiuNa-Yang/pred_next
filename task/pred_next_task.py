import pickle
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from pytorch_lightning import LightningModule
from torch.nn import functional as F
from torch.utils.data import DataLoader

from config.base_config import DataModuleConfig, TrainConfig, json_config
from dataset.pred_next_dataset import PredNextDataset
from model.base_model import PredNextModel
from model.tokenizer import Tokenizer
from utils import schedulers, metrics
from transformers import AdamW, get_cosine_schedule_with_warmup, get_linear_schedule_with_warmup


@json_config
@dataclass
class PredNextConfig(TrainConfig, DataModuleConfig):
    name: str = ''
    freeze: bool = False


class PredNextTask(LightningModule):
    def __init__(self, cfg: PredNextConfig):
        super().__init__()
        self.cfg = cfg
        self.model = PredNextModel(self.cfg)
        self.loss = nn.CrossEntropyLoss()
        self.tokenizer = Tokenizer(cfg.tokenizer_name, cfg.max_len)
        self.train_data = None
        self.val_data = None
        self.test_data = None
        with open(self.cfg.intent2index_path, 'rb') as f:
            self.intent2index = pickle.load(f)

    def forward(self, input_sent, input_intent, output_intent):

        outputs = self.model(input_ids=input_sent['input_ids'],
                             attention_mask=input_sent['attention_mask'],
                             token_type_ids=input_sent['token_type_ids'],
                             input_intent=input_intent)

        loss = self.loss(outputs, output_intent)
        return outputs, loss

    def training_step(self, batch, batch_idx):
        input_sent = batch['input_sent']
        input_intent = batch['input_intent']
        output_intent = batch['output_intent']

        outputs, loss = self(input_sent, input_intent, output_intent)
        self.log('train/loss', loss)
        self.log('lr', self.optimizers().optimizer.state_dict()['param_groups'][0]['lr'])
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        input_sent = batch['input_sent']
        input_intent = batch['input_intent']
        output_intent = batch['output_intent']

        outputs, loss = self(input_sent, input_intent, output_intent)
        self.log('val/loss', loss)
        preds = torch.argsort(outputs, dim=1, descending=True)

        metric_dict = {}
        for k in [1, 5]:
            metric_dict.update(metrics.top_k_accuracy(preds, output_intent, k=k))
        self.log_dict(metric_dict)
        return loss

    def predict_step(self, batch: dict, batch_idx: int, dataloader_idx: int = 0):
        input_sent = batch['input_sent']
        input_intent = batch['input_intent']
        outputs = self.model(input_ids=input_sent['input_ids'],
                             attention_mask=input_sent['attention_mask'],
                             token_type_ids=input_sent['token_type_ids'],
                             input_intent=input_intent)
        outputs = F.softmax(outputs, dim=1)
        scores, preds = torch.max(outputs, dim=1)
        return preds.tolist(), scores.tolist()

    def on_predict_epoch_end(self, results):
        preds, scores = [], []
        for _p, _s in results[0]:
            preds += _p
            scores += _s
        results[0] = preds, scores

    # def configure_optimizers(self):
    #     optimizer, scheduler = schedulers.get_linear_schedule_with_warmup(self.model, self.cfg, self.trainer)
    #     return [optimizer], [scheduler]

    def configure_optimizers(self):
        optimizer = AdamW(self.model.parameters(), lr=self.cfg.learning_rate)
        return [optimizer]

    def setup(self, stage: Optional[str] = None):
        print(stage)
        # if stage in ['train', 'validate']:
        train_data = PredNextDataset(self.cfg.train_data_path)
        test_data = PredNextDataset(self.cfg.test_data_path)
        val_data = PredNextDataset(self.cfg.val_data_path)

        self.train_data = train_data
        self.test_data = test_data
        self.val_data = val_data
        print(f'train data size: {len(self.train_data)}')
        print(f'val data size: {len(self.test_data)}')

    def collate_fn_predict(self, batch):
        """
        :param batch:
        :return: 用于做预测 所以没有标签
        """
        input_intent, input_sent = zip(*batch)
        input_sent = list(input_sent)
        input_sent = self.tokenizer(input_sent)

        input_intent = [self.intent2index[_] for _ in input_intent]

        input_intent = torch.LongTensor(input_intent)
        return {'input_intent': input_intent, 'input_sent': input_sent}

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
                          , collate_fn=self.collate_fn, pin_memory=False, drop_last=True
                          , num_workers=self.cfg.num_workers)

    def val_dataloader(self):
        return DataLoader(self.test_data, batch_size=self.cfg.test_batch_size, shuffle=False
                          , collate_fn=self.collate_fn, pin_memory=False, drop_last=False
                          , num_workers=self.cfg.num_workers)

    def get_predict_dataloader(self, data, batch_size):
        return DataLoader(data, batch_size=batch_size, shuffle=False, collate_fn=self.collate_fn_predict
                          , pin_memory=False, drop_last=False)

    def check_dataloader(self):
        return DataLoader(self.test_data, batch_size=2, collate_fn=self.collate_fn, shuffle=True)
