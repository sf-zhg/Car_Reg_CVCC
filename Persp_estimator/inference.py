from torch.utils.data import DataLoader
from torch.nn import Module
import torch
from tqdm import tqdm

from typing import Callable, Tuple


def inference(
    data_loader: DataLoader, model: Module, loss: Callable
) -> Tuple[float, float, float]:
    """
    Function to run inference with.

    Parameters
    ----------
    data_loader : DataLoader
        DESCRIPTION.
    model : Module
        DESCRIPTION.
    loss : Callable
        DESCRIPTION.

    Returns
    -------
    tuple of floats for total performance and single performance.

    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model.eval()

    with torch.no_grad():
        total_loss = 0.0
        total_hood_loss = 0.0
        total_backdoor_loss = 0.0
        counter = 0

        with tqdm(data_loader) as test_bar:

            for i, (x, hood_score, backdoor_score) in enumerate(test_bar, 1):

                counter += 1

                x = x.to(device).float()
                hood_score = hood_score.to(device).float()
                backdoor_score = backdoor_score.to(device).float()

                hood_score_pred, backdoor_score_pred = model(x)

                hood_score_pred = hood_score_pred.view(hood_score)
                backdoor_score_pred = backdoor_score_pred.view(backdoor_score)

                hood_loss = loss(hood_score_pred, hood_score)
                backdoor_loss = loss(backdoor_score_pred, backdoor_score)

                # lets just assume the total consists of equal parts from both
                # scores
                losses = 0.5 * (hood_loss + backdoor_loss)

                total_loss += losses.item()
                total_hood_loss += hood_loss.item()
                total_backdoor_loss += backdoor_loss.item()

                test_bar.set_postfix(
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
