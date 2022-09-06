import torch
import torch.nn as nn
from transformers import BertConfig, BertModel
from torch import Tensor


class Pooler(nn.Module):
    def __init__(self, hidden_size, output_size=None):
        super().__init__()
        if not output_size:
            output_size = hidden_size
        self.dense = nn.Linear(hidden_size, output_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states: Tensor) -> Tensor:
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class BaseModel(nn.Module):
    def __init__(self, model_name, from_pretrained=True):
        super().__init__()
        if from_pretrained:
            self.backbone = BertModel.from_pretrained(model_name, add_pooling_layer=False)
        else:
            config = BertConfig.from_pretrained(model_name)
            self.backbone = BertModel(config, add_pooling_layer=False)

    def forward(self, input_ids, attention_mask, token_type_ids):
        outputs = self.backbone(input_ids, attention_mask, token_type_ids)[0]
        pooler_output = outputs[:, 0, :]
        return outputs, pooler_output


class ModelwithPooler(BaseModel):
    def __init__(self, model_name, from_pretrained=True):
        super(ModelwithPooler, self).__init__(model_name, from_pretrained)
        self.hidden_size = self.backbone.config.hidden_size
        self.pooler = Pooler(self.hidden_size)

    def forward(self, input_ids, attention_mask, token_type_ids):
        outputs = self.backbone(input_ids, attention_mask, token_type_ids)[0]
        pooler_output = self.pooler(outputs)
        return pooler_output


class PredNextModelwithIntent(nn.Module):
    def __init__(self, model_name, cfg, from_pretrained=False):
        super().__init__()
        self.backbone = ModelwithPooler(model_name, from_pretrained=from_pretrained)

        self.intent_embedding = nn.Embedding(num_embeddings=cfg.num_embed, embedding_dim=cfg.embed_dim)
        self.fc = nn.Linear(self.backbone.hidden_size + cfg.embed_dim, cfg.num_embed)

    def forward(self, input_intent, **x):
        print(input_intent, x)
        sent_emb = self.backbone(**x)
        intent_emb = self.intent_embedding(input_intent)

        outputs = self.fc(torch.cat([intent_emb, sent_emb], dim=1))

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
    model = PredNextModelwithIntent(model_name="../pretrained_model/rbt3", cfg=cfg_1)

    # output = model(input_ids=sample['input_sent']['input_ids'],
    #                attention_mask=sample['input_sent']['attention_mask'],
    #                token_type_ids=sample['input_sent']['token_type_ids'],
    #                input_intent=sample['input_intent'])
    output = model(sample['input_intent'], **sample['input_sent'])
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
