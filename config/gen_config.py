import json
from config import data_config
import pickle


with open(data_config.intent2index_path, 'rb') as f:
    intent2index = pickle.load(f)

intent_emb_num = len(intent2index)

config_json = {
    "model_name": "PredNext",
    "train_data_path": data_config.train_data_path,
    "test_data_path": data_config.test_data_path,
    "val_data_path": data_config.dev_data_path,

    "intent2index_path": data_config.intent2index_path,
    "tokenizer_name": data_config.roberta_path,
    "bert_path": data_config.roberta_path,
    "num_embed": intent_emb_num,
    "default_root_dir": data_config.pred_next_log
}

json.dump(config_json, open("configs/test_config.json", "w"))
