import torch.nn as nn
from transformers import AutoConfig, AutoModelForSequenceClassification


class ClsModel(nn.Module):
    def __init__(self, model_name, num_labels, from_pretrained=True):
        super().__init__()
        if from_pretrained:
            self.backbone = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
        else:
            config = AutoConfig.from_pretrained(model_name, num_labels=num_labels)
            self.backbone = AutoModelForSequenceClassification.from_config(config)

    def forward(self, input_ids, attention_mask, token_type_ids):
        outputs = self.backbone(input_ids=input_ids,
                                token_type_ids=token_type_ids,
                                attention_mask=attention_mask)[0]
        return outputs
