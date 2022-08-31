from pytorch_lightning import Trainer
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR


def scheduler_warmup(model, conf):
    optimizer = AdamW([p for p in model.parameters() if p.requires_grad],
                      lr=conf.lr)

    def lr_lambda(current_step):
        if current_step < conf.warmup_steps:
            return float(current_step) / float(max(1, conf.warmup_steps))
        return 1
        # return max(0, 1 - (current_step - warmup_steps) / (all_steps - warmup_steps))

    scheduler = {
        "scheduler": LambdaLR(optimizer, lr_lambda),
        "name": "learning_rate",
        "interval": "step",
        "frequency": 1,
    }
    return optimizer, scheduler


def get_linear_schedule_with_warmup(model, config, trainer: Trainer):
    lr = config.learning_rate
    weight_decay = config.weight_decay
    warmup_steps = config.warmup_steps
    num_training_steps = trainer.estimated_stepping_batches
    # print('{num_training_steps=}')

    def lr_lambda(current_step: int):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        return max(
            0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - warmup_steps))
        )

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    scheduler = {'scheduler': LambdaLR(optimizer, lr_lambda),
                 'interval': 'step',
                 'frequency': 1}
    return optimizer, scheduler
