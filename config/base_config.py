import json
from dataclasses import dataclass
from os.path import join
from pprint import pprint
from typing import Union

from config import ROOT


def json_config(cls):
    def inner(_cls):
        _cls.from_json = classmethod(from_json)
        return _cls

    return inner(cls)


def from_json(cls, path):
    with open(path, 'r') as f:
        cls = cls(**json.load(f))

    # transfer path to absolute path
    for k, v in cls.__dict__.items():
        if v and k.endswith('_path'):
            cls.__dict__[k] = join(ROOT, v)
    pprint(cls.__dict__)
    return cls


@dataclass
class TrainConfig:
    model_name: str = "PredNext"
    epochs: int = 5
    learning_rate: float = 2e-5
    weight_decay: float = 0
    warmup_steps: int = 10
    embed_dim: int = 64
    num_embed: int = 0
    val_check_interval: Union[int, float] = 1.0
    default_root_dir: str = ''
    num_workers: int = 1


@dataclass
class DataModuleConfig:
    train_data_path: str = ''
    test_data_path: str = ''
    val_data_path: str = ''
    test_size: float = 0.1
    train_batch_size: int = 32
    test_batch_size: int = 32
    val_batch_size: int = 32

    intent2index_path: str = ''
    index2intent_path: str = ''
    tokenizer_name: str = ''
    bert_path: str = ''
    max_len: int = 64


if __name__ == "__main__":
    print(TrainConfig(epochs=1000).epochs)
