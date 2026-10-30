import torch
import torch.optim as optim
from torch.optim.lr_scheduler import _LRScheduler
import warnings
from typing import List
from .func import decay_func_2

class AlwaysConstantLR(_LRScheduler):
    def __init__(self, optimizer: optim.Optimizer, last_epoch: int=-1) -> None:
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self) -> List[float]:
        return [base_lr for base_lr in self.base_lrs]

class PolynomialDecayLR(_LRScheduler):
    def __init__(self, optimizer: optim.Optimizer, total_epochs: int, lr_min: float = 0.0, power: int = 2, last_epoch: int =-1) -> None:
        self.lr_max = optimizer.defaults['lr']
        self.lr_min = lr_min
        if self.lr_max < self.lr_min:
            raise ValueError("lr_max must be larger than lr_min")
        self.power = power
        self.total_epochs = total_epochs
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self) -> List[float]:
        cur_epoch = max(0, min(self.last_epoch, self.total_epochs - 1))
        if self.total_epochs == 1:
            scale_factor = 0
        else:
            scale_factor = (1 - cur_epoch / (self.total_epochs-1)) ** self.power
        return [self.lr_min+(self.lr_max-self.lr_min)*scale_factor
                for _ in self.optimizer.param_groups]

def get_lr_scheduler(sche_name: str, optimizer: optim.Optimizer, epochs: int=None)->_LRScheduler:
    """
    sche_name: scheduler_name we use: {constant, decay, cosine, poly}
    optimizer: optimizer we use: {sgd, adam, adamw, rmsprop}
    epochs: the total number of epochs used in training
    """
    
    if sche_name == "constant":
        lr_scheduler = AlwaysConstantLR(optimizer)
    elif sche_name == "decay":
        lr_scheduler = optim.lr_scheduler.LambdaLR(optimizer=optimizer, lr_lambda=decay_func_2)
    elif sche_name == "cosine":
        lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=0.0)
    elif sche_name == "poly":
        lr_scheduler = PolynomialDecayLR(optimizer, total_epochs=epochs)
    else:
        raise ValueError(f"Invalid scheduler name '{sche_name}'. Available options are: 'constant', 'decay', 'cosine', 'poly'.")
    return lr_scheduler
