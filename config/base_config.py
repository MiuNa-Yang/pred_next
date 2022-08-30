from dataclasses import dataclass


@dataclass
class TrainConfig:
    epochs: int = 100
    learning_rate: float = 2e-5
    weight_decay: float = 0
    warmup_steps: int = 1000


@dataclass
class DataModuleConfig:
    train_data_path: str = ''
    val_data_path: str = ''
    test_size: float = 0.1
    train_batch_size: int = 32
    val_batch_size: int = 32


if __name__ == "__main__":
    print(TrainConfig(epochs=1000).epochs)
