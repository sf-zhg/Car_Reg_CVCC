import torch
from torch.optim import Optimizer
from torch.nn import Module
from tqdm import tqdm
from torch.utils.data import DataLoader

from typing import Callable, Tuple


def trainer(
    data_loader: DataLoader,
    optimizer: Optimizer,
    loss: Callable,
    model: Module,
) -> Tuple[float, float, float]:
    """
    Trainer over one epoch.

    Parameters
    ----------
    data_loader : DataLoader
        data loader of the data.
    optimizer : Optimizer
        optimization algorithm has to be called before.
    loss : Callable
        loss function.
    model : Module
        backbone model.

    Returns
    -------
    tuple of floats
        average loss over one epoch- total and single losses for each score.
    """

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model.train()
    model.to(device)

    total_loss = 0.0
    total_hood_loss = 0.0
    total_backdoor_loss = 0.0
    counter = 0

    with tqdm(data_loader) as train_bar:

        for i, (x, hood_score, backdoor_score) in enumerate(train_bar, 1):

            counter += 1

            x = x.to(device).float()
            hood_score = hood_score.to(device).float()
            backdoor_score = backdoor_score.to(device).float()

            hood_score_pred, backdoor_score_pred = model(x)

            hood_score_pred = hood_score_pred.view(hood_score.shape)
            backdoor_score_pred = backdoor_score_pred.view(backdoor_score.shape)

            hood_loss = loss(hood_score_pred, hood_score)
            backdoor_loss = loss(backdoor_score_pred, backdoor_score)

            # lets just assume the total consists of equal parts from both
            # scores
            losses = 0.5 * (hood_loss + backdoor_loss)
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            total_loss += losses.item()
            total_hood_loss += hood_loss.item()
            total_backdoor_loss += backdoor_loss.item()

            train_bar.set_postfix(
                {
                    "Iteration": i,
                    "loss": total_loss / counter,
                    "hood loss": total_hood_loss / counter,
                    "backdoor loss": total_backdoor_loss / counter,
                }
            )

    return (
        total_loss / counter,
        total_hood_loss / counter,
        total_backdoor_loss / counter,
    )
