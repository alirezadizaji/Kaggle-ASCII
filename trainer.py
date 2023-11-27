from typing import Callable, Iterator

import torch
from torch import nn
import torch.nn.functional as F

from .base_model import BaseModel

class Trainer:
    def __init__(self, num_epochs: int, model: BaseModel, batch_size: int,
            x_train: torch.Tensor, y_train: torch.Tensor, x_val: torch.Tensor, 
            y_val: torch.Tensor, optim_initializer: torch.optim = Callable[[Iterator[torch.nn.Parameter]], torch.optim.Optimizer]) -> None:
        
        self.model = model
        self.batch_size: int = batch_size
        self.num_epochs: int = num_epochs

        self.x_train: torch.Tensor = x_train
        self.y_train: torch.Tensor = y_train
        self.x_val: torch.Tensor = x_val
        self.y_val: torch.Tensor = y_val
        self.optimizer: torch.optim.Optimizer = optim_initializer(self.model.parameters())
    
    

    def train(self):
        self.model.train()
        num_samples = self.x_train.shape[0]
        sample_indices = torch.arange(num_samples, device=self.x_train.device)

        list_batch_indices = sample_indices.split(self.batch_size)
        for e in range(self.num_epochs):
            for iter, l in enumerate(list_batch_indices):
                self.optimizer.zero_grad()
                x_train, y_train = self.x_train[l], self.y_train[l]
                x_val, y_val = self.x_val[l], self.y_val[l]

                out = self.model(x_train)
                loss = self.model.loss_calc(out, y_train)
                acc = self.model.get_acc(out, y_val)

                loss.backward()
                self.optimizer.step()
                
                if iter % 10 == 0 or iter == len(list_batch_indices) - 1:
                    self.model.eval()
                    with torch.no_grad():
                        val_out = self.model(x_val)
                        val_loss = self.model.loss_calc(val_out, y_val)
                        val_acc = self.model.get_acc(val_out, y_val)

                        print(f"Epoch {e} -->Train Loss: {loss:.4f}, Train ACC: {acc:.2%} ;;; Val Loss: {val_loss:.4f}, Val ACC: {val_acc:.2%}", flush=True)
                    

                    self.model.train()