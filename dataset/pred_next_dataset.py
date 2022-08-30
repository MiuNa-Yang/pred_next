import pandas as pd
from torch.utils.data import Dataset


class PredNextDataset(Dataset):
    def __init__(self, path):
        self.data = self.load_data(path)

    @staticmethod
    def load_data(data_path):
        data = pd.read_csv(data_path)
        return data

    def __getitem__(self, item):
        return self.data[item]

    def __len__(self):
        return len(self.data)


if __name__ == "__main__":
    print()
