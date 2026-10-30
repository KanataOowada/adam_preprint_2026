import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple
from abc import ABC, abstractmethod


class BatchScheduler(ABC):
    """
    BatchScheduler is an abstract class for updating the batch size
    """
    def __init__(self, dataset: Dataset, batch_size: int, shuffle: bool, num_workers: int, last_epoch: int = -1) -> None:
        """
        dataset (Dataset): The dataset used for training or validation.
        batch_size (int): The initial batch size to be used for DataLoader.
        shuffle (bool): Whether to shuffle the data at every epoch.
        num_workers (int): The number of subprocesses to use for data loading.
        last_epoch (int): The index of the last epoch. Default is -1, which means training starts from the beginning.
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.dataloader = DataLoader(dataset=self.dataset, batch_size=self.batch_size, shuffle=self.shuffle, num_workers=self.num_workers)
        self.last_epoch = last_epoch

    @abstractmethod
    def get_batch(self) -> Tuple[int, DataLoader]:
        pass

    def step(self) -> DataLoader:
        self.last_epoch += 1
        self.batch_size = self.get_batch()

        new_dataloader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=self.shuffle, num_workers=self.num_workers)
        self.dataloader = new_dataloader
        return self.batch_size, self.dataloader

class BatchConstantScheduler(BatchScheduler):
    def __init__(self, dataset: Dataset, batch_size: int, shuffle: bool, num_workers: int, last_epoch: int = -1) -> None:
        super().__init__(dataset, batch_size, shuffle, num_workers, last_epoch)
    
    def get_batch(self) -> int:
        return self.batch_size

class BatchExponentialScheduler(BatchScheduler):
    def __init__(self, dataset: Dataset, batch_size: int, shuffle: bool, num_workers: int, step_size: int, delta: int, last_epoch: int = -1) -> None:
        """
        step_size (int): interval of epochs after which to increment internal counter.
        delta (int): the hyperparameter used in updating batch_size(b'_m = b_0 * delta^{m}).
        """
        self.step_size = step_size
        self.delta = delta
        self.m_last_epoch = 1
        super().__init__(dataset, batch_size, shuffle, num_workers, last_epoch)

    def get_batch(self) -> int:
        if self.last_epoch == 0 or self.last_epoch % self.step_size != 0:
            return self.batch_size
        else:
            self.m_last_epoch += 1
            self.batch_size = self.delta * self.batch_size
            return self.batch_size

def get_batch_scheduler(sche_name: str, dataset: Dataset, batch_size: int=256, shuffle: bool=True, num_workers: int=2, step_size: int=None)-> BatchScheduler:
    """
    sche_name (str): scheduler_name we use: {constant, exp}
    dataset (Dataset): The dataset used for training or validation.
    batch_size (int): The initial batch size to be used for DataLoader.
    shuffle (bool): Whether to shuffle the data at every epoch.
    num_workers (int): The number of subprocesses to use for data loading.
    step_size (int): interval of epochs after which to increment internal counter.
    """
    if sche_name == "constant":
        batch_scheduler = BatchConstantScheduler(dataset, batch_size, shuffle, num_workers)
    elif sche_name == "exp":
        batch_scheduler = BatchExponentialScheduler(dataset, batch_size, shuffle, num_workers, step_size, delta=2)
    else:
        raise ValueError(f"Invalid scheduler name '{sche_name}'. Available options are: 'constant', 'exp'.")
    return batch_scheduler
