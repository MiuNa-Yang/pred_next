from collections import defaultdict

import pandas as pd

# from conf import label_conf


class Label:
    """
    idx             序号
    lid             产品线内序号

    line_id         lineId
    int             intention/intentionName
    int_en_flag     intentionEnFlag
    iid             intentionId
    """

    def __init__(self, path):

        self.idx2iid = None
        self.idx2line_id = None
        self.idx2int = None
        self.idx2int_en_flag = None
        self.idx2lidx = None

        self.iid2idx = None
        self.int2idx = None
        self.int_en_flag2idx = None
        self.lidx2idx = None

        self.inline_idx_dict = {}
        self.mask_dict = {}

        self.df = self.load_data(path)
        self.process()
#         self.gen_line_mask()
        self.gen_inline_idx_dict()
        self.inline_label_length = {k: len(v) for k, v in self.inline_idx_dict.items()}

    def load_data(self, path):
        df = pd.read_csv(path)
        df = df.fillna('')
        idx2iid = df['intentionid'].tolist()
        idx2line_id = df['lineid'].tolist()
        idx2int_en_flag = df['intentionenflag'].tolist()
        idx2int = df['intentionname'].tolist()
        idx2lidx = df['lidx'].tolist()
        self.idx2iid = idx2iid
        self.idx2line_id = idx2line_id
        self.idx2int_en_flag = idx2int_en_flag
        self.idx2int = idx2int
        self.idx2lidx = idx2lidx
        return df

    def process(self):
        iid2idx = {v: k for k, v in enumerate(self.idx2iid)}
        int2idx = {(_int, line_id): idx
                   for idx, (_int, line_id) in enumerate(zip(self.idx2int, self.idx2line_id))}
        int_en_flag2idx = {(int_en, line_id): idx
                           for idx, (int_en, line_id) in enumerate(zip(self.idx2int_en_flag, self.idx2line_id))}
        lidx2idx = {(lidx, line_id): idx
                    for idx, (lidx, line_id) in enumerate(zip(self.idx2lidx, self.idx2line_id))}
        self.iid2idx = iid2idx
        self.int2idx = int2idx
        self.int_en_flag2idx = int_en_flag2idx
        self.lidx2idx = lidx2idx

#     def gen_line_mask(self):
#         import torch

#         line_id_mask_dict = {}
#         # line id mask
#         for line_id in self.line_id2line:
#             mask = self.df['lineId'].apply(lambda x: 1 if x == line_id else 0).tolist()
#             line_id_mask_dict[line_id] = torch.Tensor(mask)
#         # status mask
#         # status_mask = self._df['status'].apply(lambda x: 1 if x == 1 else 0).tolist()
#         # status_mask = torch.Tensor(status_mask)
#         # self.mask_dict = {k: status_mask * v for k, v in line_id_mask_dict.items()}
#         self.mask_dict = line_id_mask_dict

    def gen_inline_idx_dict(self):
        d = defaultdict(lambda: {0: 0})
        for (lidx, line_id), idx in self.lidx2idx.items():
            d[line_id][lidx] = idx
        self.inline_idx_dict = d

    def __len__(self):
        return len(self.idx2int)


