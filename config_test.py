from dataclasses import dataclass
from utils.load_config import json_config
from config.base_config import TrainConfig, DataModuleConfig


@json_config
@dataclass
class PredNextTaskConfig(TrainConfig, DataModuleConfig):
    name: str = ''
    freeze: bool = False


if __name__ == "__main__":
    cfg = PredNextTaskConfig.from_json("config/configs/config1.json")
    print(cfg.train_data_path)
