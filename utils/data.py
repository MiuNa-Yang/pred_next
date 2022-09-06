from collections import defaultdict

import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import random_split


def random_split_by_ratio(data, test_size):
    length = len(data)
    test_size = int(test_size * length)
    train_size = length - test_size
    train_data, val_data = random_split(data, [train_size, test_size],
                                        generator=torch.Generator().manual_seed(42))
    return train_data, val_data


def random_split_by_label(x, y, test_size):
    data_dict = defaultdict(list)
    for _x, _y in zip(x, y):
        data_dict[_y].append(_x)

    train_data, test_data = [], []
    for _y, x_list in data_dict.items():
        if len(x_list) <= 1:
            print(f'label {_y} removed')
            continue
        x_train_list, x_test_list = train_test_split(x_list, test_size=test_size, random_state=42)
        train_data += [[_x, _y] for _x in x_train_list]
        test_data += [[_x, _y] for _x in x_test_list]

    train_x, train_y = zip(*train_data)
    test_x, test_y = zip(*test_data)
    return list(train_x), list(train_y), list(test_x), list(test_y)


def split_data(data, test_size):
    length = len(data)
    test_size = int(test_size * length)
    train_size = length - test_size
    train_data, val_data = random_split(data, [train_size, test_size],
                                        generator=torch.Generator().manual_seed(42))
    return train_data, val_data
