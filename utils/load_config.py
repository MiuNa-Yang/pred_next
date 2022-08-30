import json
from os.path import join
from pprint import pprint
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
