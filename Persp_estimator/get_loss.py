import torch.nn as nn

from typing import Callable


def get_loss(loss: str) -> Callable:
    '''
    Return a loss function for optimization based on a input category.

    Parameters
    ----------
    loss : str
        loss specification.

    Raises
    ------
    ValueError
        DESCRIPTION.

    Returns
    -------
    Callable
        loss function from torch.nn.

    '''
    if loss == "l1":
        loss_function = nn.L1Loss()
    elif loss == "l2":
        loss_function = nn.MSELoss()
    elif loss == "huber":
        loss_function = nn.HuberLoss()
    else:
        raise ValueError("loss not implemented, you fucked up big time")

    return loss_function
