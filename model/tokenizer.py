from transformers import AutoTokenizer


class Tokenizer:
    def __init__(self, path, return_tensors='pt', model_max_length=512):
        self.return_tensors = return_tensors
        self.t = AutoTokenizer.from_pretrained(path)
        self.t.model_max_length = model_max_length

    def __call__(self, x):
        return self.t(x, return_tensors=self.return_tensors, padding=True, truncation=True)
