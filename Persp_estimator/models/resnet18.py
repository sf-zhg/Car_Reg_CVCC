import torch
import torch.nn as nn
from torchvision.models import resnet18

from typing import Tuple


class Resnet18(nn.Module):
    """
    A custom ResNet-18 model with additional layers for regression tasks.

    This model modifies the standard ResNet-18 architecture to include
    additional layers for regression. It is designed to predict scores
    for specific targets such as hood score and backdoor score.
    """    
    def __init__(self, feature_dim: int, drop_out_rate: float):
        '''
        The Initialization for resnet18.

        Parameters
        ----------
        feature_dim : int
            dims of MLP head.
        drop_out_rate : float
            drop out rate.

        Returns
        -------
        None.
        '''
        super(Resnet18, self).__init__()

        self.feature_dim = feature_dim
        self.drop_out_rate = drop_out_rate

        # Base encoder using modules from resnet18, excluding Linear and MaxPool2d
        self.f = nn.Sequential(
            *[module for name, module in resnet18().named_children()
               if not (isinstance(module, nn.Linear) or isinstance(module, nn.MaxPool2d))]
        )

        # Projection heads for regression
        self.dim = 512  # Dimension of last layer in encoder

        # 2 layer MLP head for first target
        self.head1 = nn.Sequential(
            nn.Linear(self.dim, self.feature_dim, bias=False),
            nn.BatchNorm1d(self.feature_dim),
            nn.Dropout(self.drop_out_rate),
            nn.ReLU(inplace=True),
            nn.Linear(self.feature_dim, 1, bias=True),
            nn.Sigmoid(),
        )

        # 2 layer MLP head for second target
        self.head2 = nn.Sequential(
            nn.Linear(self.dim, self.feature_dim, bias=False),
            nn.BatchNorm1d(self.feature_dim),
            nn.Dropout(self.drop_out_rate),
            nn.ReLU(inplace=True),
            nn.Linear(self.feature_dim, 1, bias=True),
            nn.Sigmoid(),
        )

        # Dropout layer for additional regularization
        self.drop_out_layer = nn.Sequential(nn.Dropout(self.drop_out_rate))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        '''
        The forward pass for resnet.

        Parameters
        ----------
        x : torch.Tensor
            image input [B, H, W, C].

        Returns
        -------
        h : Tensor
            prediction for first target score.
        g : Tensor
            prediction for second target score.
        '''
        x = self.f(x)
        features = torch.flatten(x, start_dim=1)
        x = self.drop_out_layer(features)
        h = self.head1(x)
        g = self.head2(x)

        # We typically return representations (features) as well, but not used here
        return h, g
