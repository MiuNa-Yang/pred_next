import pandas as pd
from torch.utils.data import Dataset


class PredNextDataset(Dataset):
    def __init__(self, path):
        self.data = self.load_data(path)

    @staticmethod
    def load_data(data_path):
        """

        :param data_path:
        :return: data[0]: input_intent
                 data[1]: input_sent
                 data[2]: output_intent

                 ------------------------
                 data[0]: product_line
                 data[1]: input_intent
                 data[2]: input_sent
                 data[3]: output_intent
                 ------------------------

        """
        print(data_path)
        df = pd.read_json(data_path)
        input_intent = df['input_intent']
        input_sent = df['input_sen']
        output_label = df['output_intent']
        data = list(zip(input_intent, input_sent, output_label))

        return data

    def __getitem__(self, item):
        return self.data[item]

    def __len__(self):
        return len(self.data)


if __name__ == "__main__":
    print()
