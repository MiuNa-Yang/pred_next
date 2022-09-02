import torch
import torch.nn as nn
from transformers import BertConfig, BertModel


class PredNextModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.bert_config = BertConfig.from_pretrained(cfg.bert_path, output_hidden_states=True)
        self.bert_layer = BertModel(config=self.bert_config)
        self.intent_embedding = nn.Embedding(num_embeddings=cfg.num_embed, embedding_dim=cfg.embed_dim)
        self.fc = nn.Linear(cfg.max_len + cfg.embed_dim, cfg.num_embed)
        self.sent_embed = nn.Linear(2304, 1)

    def forward(self, input_ids, attention_mask, token_type_ids, input_intent):
        bert_outputs = self.bert_layer(input_ids=input_ids,
                                       token_type_ids=token_type_ids,
                                       attention_mask=attention_mask)
        bert_layer_1, bert_layer_2, bert_layer_12 =\
            bert_outputs.hidden_states[0], bert_outputs.hidden_states[1], bert_outputs.hidden_states[-1]

        bert_output_merge = torch.cat([bert_layer_1, bert_layer_2, bert_layer_12], dim=2)

        node_emb = self.intent_embedding(input_intent)
        sent_emb2 = self.sent_embed(bert_output_merge).mean(dim=2)
        outputs = self.fc(torch.cat([sent_emb2, node_emb], dim=1))

        return outputs


if __name__ == "__main__":
    from config.base_config import DataModuleConfig, TrainConfig, json_config
    from config import data_config
    import pickle
    from dataclasses import dataclass
    from dataset.dataloader_test import PredNextDataModule

    with open(data_config.intent2index_path, 'rb') as f:
        intent2index = pickle.load(f)

    intent_emb_num = len(intent2index)


    @json_config
    @dataclass
    class PredNextConfig(TrainConfig, DataModuleConfig):
        name: str = ''
        freeze: bool = False
        num_embed = intent_emb_num


    cfg_1 = PredNextConfig.from_json("../config/configs/test_config.json")
    dm = PredNextDataModule(cfg_1)
    dm.setup()
    dl = dm.check_dataloader()
    sample = next(iter(dl))
    model = PredNextModel(cfg_1)

    output = model(input_ids=sample['input_sent']['input_ids'],
                   attention_mask=sample['input_sent']['attention_mask'],
                   token_type_ids=sample['input_sent']['token_type_ids'],
                   input_intent=sample['input_intent'])
    print(output.shape)
    topk_ids = torch.topk(output, k=4, dim=1)[1]
    print(topk_ids)
    topk_min = torch.topk(output, k=4, dim=1)[0].min(dim=1)[0].reshape(2, 1)
    mask = output >= topk_min
    # print(mask * output)
    softmax_fn = nn.Softmax(dim=1)
    print(softmax_fn(output).shape)
    label = torch.zeros(2, 194)
    label[0, 1] = 1
    label[1, 2] = 1
    # print(label)
    # print(nn.CrossEntropyLoss()(output, label))
