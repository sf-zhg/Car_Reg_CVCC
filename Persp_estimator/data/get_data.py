from Persp_estimator.data.car_persp import CarPerspective
from Persp_estimator.data.get_transform import get_transform

from typing import Optional, Callable, Tuple, Any


def get_data(
    dataset: str,
    split_ratio: float,
    root: str,
    transform: Optional[Callable] = None,
    target_transform: Optional[Callable] = None,
) -> Tuple[Any, Any]:
    """
    Function to get dataset.

    Parameters
    ----------
    dataset : str
        name of dataset.
    split : float
        test or train set.
    tranform : Optional[Callable], optional
        image transformation. The default is None.
    target_transform : Optional[Callable], optional
        target transformation. The default is None.
    split_ratio : float
        ratio of train set from whole set.
    root : str
        directory of data. should be Persp_estimator/raw_data

    Returns
    -------
    tuple of train and test set.

    """
    # define transformation functions
    train_transform = get_transform(dataset=dataset, split='train')
    test_transform = get_transform(dataset=dataset, split='test')

    if dataset == 'car_persp':
        # define train and test args (as supervised we dont need memory args)
        # usually the args can be outside the case but when handling a lot
        # of datasets it seems smart to not write every dataset calss by hand
        # which leads to different args and specification
        
        train_data = CarPerspective(root=root,
                                    split='train',
                                    plit_ratio=0.5,
                                    transform=train_transform)
        
        test_data = CarPerspective(root=root,
                                    split='test',
                                    plit_ratio=0.5,
                                    transform=test_transform)

        return train_data, test_data
    else:
        raise AttributeError('dataset not defined choose a different one')
