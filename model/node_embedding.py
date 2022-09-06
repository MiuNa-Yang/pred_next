import networkx as nx
import numpy as np
import random
import pandas as pd
from node2vec import Node2Vec
import pickle


class NodeEmbedding(object):
    def __init__(self, train_data_path):

        self.G = self.load_graph(train_data_path)

        self.dim_size = 128
        self.p = 1
        self.q = 2
        self.walk_length = 8
        self.num_walks = 600

        self.model = self.train_node2vec()

        with open('../data/resources/index2intent.pickle', 'rb') as f:
            self.intent2index = pickle.load(f)
        self.node_embedding = self.get_embedding()

    @staticmethod
    def load_graph(path):
        train_df = pd.read_csv(path)
        df_node_w = train_df[["input_intent", "output_intent"]]
        df_node_w['cnt'] = df_node_w.groupby(["input_intent", "output_intent"])['output_intent'].transform("count")
        df_node_w = df_node_w.drop_duplicates(subset=["input_intent", "output_intent"]).reset_index(drop=True)
        node_s = df_node_w["input_intent"].tolist()
        node_e = df_node_w["output_intent"].tolist()
        edge_w = df_node_w["cnt"].tolist()
        G = nx.DiGraph()
        for i in range(len(node_s)):
            G.add_edge(node_s[i], node_e[i], weight=edge_w[i])

        return G

    def train_node2vec(self, workers=1, window=1):
        node2vec = Node2Vec(self.G,
                            dimensions=self.dim_size,  # 嵌入维度
                            p=self.p,  # 回家参数
                            q=self.q,  # 外出参数
                            walk_length=self.walk_length,  # 随机游走最大长度
                            num_walks=self.num_walks,  # 每个节点作为起始节点生成的随机游走个数
                            workers=workers  # 并行线程数
                            )

        model = node2vec.fit(window=window)
        return model

    def get_embedding(self):
        idx2emb = np.random.rand(1, self.dim_size)
        for intent in self.intent2index:
            idx2emb = np.vstack((idx2emb, self.model.wv.get_vector(intent)))
        idx2emb = idx2emb[1:]
        print(idx2emb)
        return idx2emb


if __name__ == "__main__":
    ne = NodeEmbedding("../data/processed_data/train_data_v3.csv")
    print(ne.node_embedding.shape)


