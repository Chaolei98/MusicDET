import math
from torch.optim.lr_scheduler import LambdaLR
from functools import partial


def get_scheduler(
    optimizer,
    start_lr,
    max_lr,
    min_lr,
    warmup_epochs,
    sustain_epochs,
    total_epochs,
    decay,
    mode="cosine",
):
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (max_lr - start_lr) / warmup_epochs * epoch + start_lr

        elif epoch < warmup_epochs + sustain_epochs:
            return max_lr

        elif mode == "exponential":
            return (max_lr - min_lr) * decay ** (
                epoch - warmup_epochs - sustain_epochs
            ) + min_lr

        elif mode == "step":
            return max_lr * decay ** ((epoch - warmup_epochs - sustain_epochs) // 2)

        elif mode == "cosine":
            decay_total_epochs = total_epochs - warmup_epochs - sustain_epochs + 3
            decay_epoch_index = epoch - warmup_epochs - sustain_epochs
            phase = math.pi * decay_epoch_index / decay_total_epochs
            cosine_decay = 0.5 * (1 + math.cos(phase))
            return (max_lr - min_lr) * cosine_decay + min_lr

        else:
            raise ValueError(
                f"Unsupported mode '{mode}'. Supported modes are 'exp', 'step', 'cosine'."
            )

    return LambdaLR(optimizer, lr_lambda)

def _get_cosine_schedule_with_warmup_lr_lambda(
    current_step: int,
    *,
    num_warmup_steps: int,
    num_training_steps: int,
    num_cycles: float,
):
    if current_step < num_warmup_steps:
        return float(current_step) / float(max(1, num_warmup_steps))
    progress = float(current_step - num_warmup_steps) / float(
        max(1, num_training_steps - num_warmup_steps)
    )
    return max(
        0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress))
    )

def get_cosine_schedule_with_warmup(
    optimizer, num_warmup_steps, num_training_steps, num_cycles=0.5, last_epoch=-1
):
    """
    Create a schedule with a learning rate that decreases following the values of the cosine function between the
    initial lr set in the optimizer to 0, after a warmup period during which it increases linearly between 0 and the
    initial lr set in the optimizer.

    Args:
        optimizer ([`~torch.optim.Optimizer`]):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (`int`):
            The number of steps for the warmup phase.
        num_training_steps (`int`):
            The total number of training steps.
        num_cycles (`float`, *optional*, defaults to 0.5):
            The number of waves in the cosine schedule (the defaults is to just decrease from the max value to 0
            following a half-cosine).
        last_epoch (`int`, *optional*, defaults to -1):
            The index of the last epoch when resuming training.

    Return:
        `torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """

    lr_lambda = partial(
        _get_cosine_schedule_with_warmup_lr_lambda,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
        num_cycles=num_cycles,
    )
    return LambdaLR(optimizer, lr_lambda, last_epoch)
