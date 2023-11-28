from typing import Callable, Iterator

import torch
from torch import nn
import torch.nn.functional as F

from .base_model import BaseModel

class Trainer:
    def __init__(self, num_epochs: int, model: BaseModel, batch_size: int,
            optim_initializer: torch.optim = Callable[[Iterator[torch.nn.Parameter]], torch.optim.Optimizer]) -> None:
        
        self.model = model
        self.batch_size: int = batch_size
        self.num_epochs: int = num_epochs

        self.optimizer: torch.optim.Optimizer = optim_initializer(self.model.parameters())

        self.best_model_param: Iterator[torch.nn.Parameter] = None
        self.best_val_acc = -torch.inf
    
    def _train_for_one_epoch(self, x_train: torch.Tensor, y_train: torch.Tensor) -> None:
        num_samples = x_train.shape[0]
        sample_indices = torch.randperm(num_samples, device=x_train.device)
        list_batch_indices = sample_indices.split(self.batch_size)

        for iter, l in enumerate(list_batch_indices):
            self.optimizer.zero_grad()
            xb_train, yb_train = x_train[l], y_train[l]

            out = self.model(xb_train)
            loss = self.model.loss_calc(out, yb_train)
            acc = self.model.get_acc(out, yb_train)

            loss.backward()
            self.optimizer.step()
            
            if iter % 10 == 0 or iter == len(list_batch_indices) - 1:
                print(f"\tEpoch {self.e} Iteration {iter} -->Train Loss: {loss:.4f}, Train ACC: {acc:.2%}", flush=True)
    
    
    def _eval_for_one_epoch(self, x_val: torch.Tensor, y_val: torch.Tensor) -> None:
        num_samples = x_val.shape[0]
        sample_indices = torch.randperm(num_samples, device=x_val.device)
        list_batch_indices = sample_indices.split(self.batch_size)
        
        losses = []
        accs = []
        for l in list_batch_indices:
            xb_val, yb_val = x_val[l], y_val[l]
            
            out = self.model(xb_val)
            loss = self.model.loss_calc(out, yb_val)
            acc = self.model.get_acc(out, yb_val)

            losses.append(loss.item())
            accs.append(acc.item())

        mloss = torch.mean(losses).item()
        maccs = torch.mean(accs).item()
        
        print(f"@@ Epoch {self.e} -->Val Loss: {mloss:.4f}, Val ACC: {maccs:.2%}", flush=True)

        if maccs >= self.best_val_acc:
            self.best_val_acc = maccs
            self.best_model_param = self.model.parameters()

    def train(self,  x_train: torch.Tensor, y_train: torch.Tensor, 
        x_val: torch.Tensor, y_val: torch.Tensor) -> None:
        """Train the model on given data and evaluate it periodically."""
        self.model.train()

        for self.e in range(self.num_epochs):
            self.model.train()
            self._train_for_one_epoch(x_train, y_train)
            
            self.model.eval()
            with torch.no_grad():
                self._eval_for_one_epoch(x_val, y_val)


    def evaluate(self, x_test: torch.Tensor) -> torch.Tensor:
        self.model.eval()
        with torch.no_grad():
            out = self.model(x_test)
        
        return out.argmax(1)