import torch
import torch.nn as nn
from torchvision.models import resnet18

from typing import Tuple


class Resnet18(nn.Module):
    def __init__(self, feature_dim: int, drop_out_rate: float):
        super(Resnet18, self).__init__()

        self.feature_dim = feature_dim
        self.drop_out_rate = drop_out_rate

        # initialize resnet. no time and need to write it from scratch, but
        # best practice would be to define a base class and write the models
        # into the base class
        self.f = []
        for name, module in resnet18().named_children():
            if name == "conv1":
                module = nn.Conv2d(
                    3, 64, kernel_size=3, stride=1, padding=1, bias=False
                )
            # we need ti drop the last layers form resnet
            if not isinstance(module, nn.Linear) and not isinstance(
                module, nn.MaxPool2d
            ):
                self.f.append(module)
        # base encoder
        self.f = nn.Sequential(*self.f)
        # projection heads for regression
        # dim of last layer in f:
        self.dim = 512
        # 2 layer MLP head, should also be adaptable but difficult to fit
        # everything in 3h time, hence the parametrization is identical
        self.g = nn.Sequential(
            nn.Linear(self.dim, self.feature_dim, bias=False),
            nn.BatchNorm1d(self.feature_dim),
            nn.Dropout(self.drop_out_rate),
            nn.ReLU(inplace=True),
            nn.Linear(self.feature_dim, 1, bias=True),
            nn.Sigmoid(),
        )

        # second head for second target
        self.h = nn.Sequential(
            nn.Linear(self.dim, self.feature_dim, bias=False),
            nn.BatchNorm1d(self.feature_dim),
            nn.Dropout(self.drop_out_rate),
            nn.ReLU(inplace=True),
            nn.Linear(self.feature_dim, 1, bias=True),
            nn.Sigmoid(),
        )
        # drop out for additional regularization
        self.drop_out_layer = nn.Sequential(nn.Dropout(self.drop_out_rate))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.f(x)
        features = torch.flatten(x, start_dim=1)
        x = self.drop_out_layer(features)
        h = self.h(x)
        g = self.g(x)

        # we should normally also return the representations features but in
        # this case it is not needed
        return h, g
