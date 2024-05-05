from torch.nn import Module

from Persp_estimator.models.resnet18 import Resnet18


def get_model(arch: str, feature_dim: int, drop_out_rate: float) -> Module:
    """
    The function is to access a backbone.

    Parameters
    ----------
    arch : str
        architecture like resnet18.
    feature_dim : int
        feature dimensions of MLP heads.
    drop_out_rate : float
        rat of dropping neurons.

    Returns
    -------
    Module
        nn.Module class.
    """

    if arch == 'resnet18':
        return Resnet18(feature_dim=feature_dim, drop_out_rate=drop_out_rate)
    else:
        raise AttributeError(f"{arch} backbonenot defined, choose another one")
