import torch
import torch.optim as optim
from typing import List, Callable
from abc import ABC, abstractmethod

class Beta2Scheduler(ABC):
    """
        Beta2Scheduler is an abstract class for updating the momentum optimizer's beta value。
    """
    def __init__(self, optimizer: optim.Optimizer, last_epoch: int = -1) -> None:
        """
        optimizer (optim.Optimizer): An optimizer with a beta_1 parameter
        last_epoch (int): The current epoch number
        """
        self.optimizer = optimizer
        self.last_epoch = last_epoch
        self.base_beta_2 = [group['betas'][1] for group in optimizer.param_groups]

    @abstractmethod
    def get_beta_2(self) -> List[float]:
        pass

    def step(self) -> float:
        self.last_epoch += 1
        new_beta_2 = self.get_beta_2()
        for param_group, beta_2 in zip(self.optimizer.param_groups, new_beta_2):
            beta_1 = param_group['betas'][0]
            param_group['betas'] = (beta_1, beta_2)
        return new_beta_2[0]
    
    def state_dict(self) -> dict:
        return {
			'last_epoch': self.last_epoch,
			'base_beta_2': self.base_beta_2,
		}

    def load_state_dict(self, state_dict: dict) -> None:
        self.last_epoch = state_dict['last_epoch']
        self.base_beta_2 = state_dict['base_beta_2']


class Beta2ConstantScheduler(Beta2Scheduler):
    def __init__(self, optimizer: optim.Optimizer, last_epoch: int = -1) -> None:
        super().__init__(optimizer, last_epoch)
        self.constant_beta_2 = self.base_beta_2[0]

    def get_beta_2(self) -> List[float]:
        return [self.constant_beta_2 for _ in self.optimizer.param_groups]

def get_beta_2_scheduler(sche_name: str, optimizer: optim.Optimizer, step_size: int=None)-> Beta2Scheduler:
    """
    sche_name (str): scheduler_name we use: {constant}
    optimizer (optim.Optimizer): optimizer we use: {sgd, adam, adamw, rmsprop}
    step_size (int): interval of epochs after which to increment internal counter
    """
    if sche_name == "constant":
        beta_2_scheduler = Beta2ConstantScheduler(optimizer)
    else:
        raise ValueError(f"Invalid scheduler name '{sche_name}'. Available options are: 'constant'.")
    return beta_2_scheduler
