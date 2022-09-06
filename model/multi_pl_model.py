import torch
import torch.nn as nn
from transformers import BertConfig, BertModel
from torch import Tensor


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


class ModelwithPooler(BaseModel):
    def __init__(self, model_name, from_pretrained=True):
        super(ModelwithPooler, self).__init__(model_name, from_pretrained)
        self.hidden_size = self.backbone.config.hidden_size
        self.pooler = Pooler(self.hidden_size)

    def forward(self, input_ids, attention_mask, token_type_ids):
        outputs = self.backbone(input_ids, attention_mask, token_type_ids)[0]
        pooler_output = self.pooler(outputs)
        # print(pooler_output.shape)
        return pooler_output


class MultiTaskClsModel(nn.Module):
    def __init__(self, model_name, class_num_dict, from_pretrained=False):
        super().__init__()
        self.backbone = ModelwithPooler(model_name, from_pretrained=from_pretrained)
        self.hidden_size = self.backbone.hidden_size
        self.dropout = nn.Dropout(0.1)
        self.cls = nn.ModuleDict()
        self.intent_embeddings = nn.ModuleDict()
        for task_id, class_num in class_num_dict.items():
            self.cls[str(task_id)] = nn.Linear(self.hidden_size + 128, class_num)
            self.intent_embeddings[str(task_id)] = nn.Embedding(class_num, 128)

    def forward(self, task_id, input_idx, **x):
        sent_emb = self.backbone(**x)
        intent_emb = self.intent_embeddings[str(task_id)](input_idx)
        z = torch.cat([intent_emb, sent_emb], dim=1)
        z = self.dropout(z)
        outputs = self.cls[str(task_id)](z)
        return outputs

