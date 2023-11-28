from abc import ABC, abstractmethod
from torch import nn
import torch

class BaseModel(nn.Module, ABC):

    @abstractmethod
    def forward(self, x_train: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def loss_calc(self, out: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        pass

    def get_acc(self, out: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        pred = out.argmax(1)
        return (pred == y).float().mean()