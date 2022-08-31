from os.path import join

from config import DATA_DIR, ROOT


processed_dir = join(DATA_DIR, "processed_data")
resources_dir = join(DATA_DIR, "resources")
pretrained_model_dir = join(ROOT, "pretrained_model")
log_dir = join(ROOT, "log")

test_data_path = join(processed_dir, "test_data_v3.json")
train_data_path = join(processed_dir, "train_data_v3.json")
dev_data_path = join(processed_dir, "dev_data_v3.json")

intent2index_path = join(resources_dir, "intent2index.pickle")
index2intent_path = join(resources_dir, "index2intent.pickle")

roberta_path = join(pretrained_model_dir, "roberta")
pred_next_log = join(log_dir, "pred_next")
