from os.path import join

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

from config import ROOT
from task.pred_next_task import PredNextConfig, PredNextTask


if __name__ == "__main__":

    train_cfg = PredNextConfig.from_json("../config/configs/test_config.json")

    task = PredNextTask(train_cfg)

    model_output_name = f"{train_cfg.model_name.split('/')[-1]}_{train_cfg.name}_cls"

    checkpoint_callback = ModelCheckpoint(dirpath=join(ROOT, 'outputs', 'models'),
                                          filename=model_output_name,
                                          save_top_k=1, monitor='val/loss', mode='min', save_weights_only=True)

    logger = TensorBoardLogger(train_cfg.default_root_dir, name=model_output_name)

    trainer = Trainer(
        default_root_dir=train_cfg.default_root_dir,
        logger=logger,
        callbacks=checkpoint_callback,
        # gpus=2,
        precision=16,
        max_epochs=train_cfg.epochs,
        num_sanity_val_steps=0
        # val_check_interval=train_cfg.val_check_interval,
    )

    # trainer.validate(task)
    trainer.fit(task)
