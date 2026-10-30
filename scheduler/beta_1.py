import torch
import torch.optim as optim
from typing import List, Callable
from abc import ABC, abstractmethod

class Beta1Scheduler(ABC):
    """
        Beta1Scheduler is an abstract class for updating the momentum optimizer's beta value。
    """
    def __init__(self, optimizer: optim.Optimizer, last_epoch: int = -1) -> None:
        """
        optimizer (optim.Optimizer): An optimizer with a beta_1 parameter
        last_epoch (int): The current epoch number
        """
        self.optimizer = optimizer
        self.last_epoch = last_epoch
        self.base_beta_1 = [group['betas'][0] for group in optimizer.param_groups]

    @abstractmethod
    def get_beta_1(self) -> List[float]:
        pass

    def step(self) -> float:
        self.last_epoch += 1
        new_beta_1 = self.get_beta_1()
        for param_group, beta_1 in zip(self.optimizer.param_groups, new_beta_1):
            beta_2 = param_group['betas'][1]
            param_group['betas'] = (beta_1, beta_2)
        return new_beta_1[0]
    
    def state_dict(self) -> dict:
        return {
			'last_epoch': self.last_epoch,
			'base_beta_1': self.base_beta_1,
		}

    def load_state_dict(self, state_dict: dict) -> None:
        self.last_epoch = state_dict['last_epoch']
        self.base_beta_1 = state_dict['base_beta_1']


class Beta1ConstantScheduler(Beta1Scheduler):
    def __init__(self, optimizer: optim.Optimizer, last_epoch: int = -1) -> None:
        super().__init__(optimizer, last_epoch)
        self.constant_beta_1 = self.base_beta_1[0]

    def get_beta_1(self) -> List[float]:
        return [self.constant_beta_1 for _ in self.optimizer.param_groups]

class Beta1StepDecayScheduler(Beta1Scheduler):
    def __init__(self, optimizer: optim.Optimizer, step_size: int, eta: float= 0.1, last_epoch: int = -1) -> None:
        self.step_size = step_size
        self.eta = eta
        super().__init__(optimizer, last_epoch)
    
    def get_beta_1(self) -> List[float]:
        if self.last_epoch == 0 or self.last_epoch % self.step_size != 0:
            return [group['betas'][0] for group in self.optimizer.param_groups]
        return [group['betas'][0] * self.eta for group in self.optimizer.param_groups]

def get_beta_1_scheduler(sche_name: str, optimizer: optim.Optimizer, step_size: int=None)-> Beta1Scheduler:
    """
    sche_name (str): scheduler_name we use: {constant, step_decay}
    optimizer (optim.Optimizer): optimizer we use: {sgd, adam, adamw, rmsprop}
    step_size (int): interval of epochs after which to increment internal counter
    """
    if sche_name == "constant":
        beta_1_scheduler = Beta1ConstantScheduler(optimizer)
    elif sche_name == "step_decay":
        beta_1_scheduler = Beta1StepDecayScheduler(optimizer, step_size=step_size, eta=0.5)
    else:
        raise ValueError(f"Invalid scheduler name '{sche_name}'. Available options are: 'constant', 'step_decay'.")
    return beta_1_scheduler
