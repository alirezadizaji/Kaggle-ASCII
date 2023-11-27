from typing import List

import torch
from torch import nn

from .res_block import ResBlock
from .base_model import BaseModel

class ResNet(BaseModel):
    def __init__(self, in_channels: int, out_channels: List[int], in_linear: int, out_linear: int,
                  num_classes: int, dropout: float = 0.5) -> None:
        super().__init__()

        self.res_blocks = nn.ModuleList([ResBlock(in_channels, out_channels[0])])
        self.pool = nn.ModuleList([])

        for i, o in enumerate(out_channels[:-1], out_channels[1:]):
            self.res_blocks.append(ResBlock(i, o))
            self.pool.append(nn.MaxPool2d((2, 2)))
        
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(in_linear, out_linear)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout)
        self.fc2 = nn.Linear(out_linear, num_classes)

        self.log_softmax = nn.LogSoftmax(dim=1)
        
    def forward(self, x_train: torch.Tensor) -> torch.Tensor:
        x = x_train
        for block, pool in enumerate(self.res_blocks, self.pool):
            x = block(x)
            x = pool(x)
        
        x = self.avg_pool(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)

        return x
    
    def loss_calc(self, out: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        log_out = self.log_softmax(out)
        sample_inds = torch.arange(y.numel())

        # cross entropy loss
        cross_entropy = -torch.mean(log_out[sample_inds, y.long()])
        return cross_entropy