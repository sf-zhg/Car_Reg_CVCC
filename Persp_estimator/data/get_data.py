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
        Name of dataset.
    split_ratio : float
        Ratio of train set from the whole dataset.
    root : str
        Directory of data, typically should be Persp_estimator/raw_data.
    transform : Optional[Callable], optional
        Image transformation function. Default is None.
    target_transform : Optional[Callable], optional
        Target transformation function. Default is None.

    Returns
    -------
    tuple
        A tuple containing the training set and the test set.
    """
    # Define transformation functions
    train_transform = get_transform(dataset=dataset, split='train')
    test_transform = get_transform(dataset=dataset, split='test')

    if dataset == 'car_persp':
        # Define train and test data with correct split ratio
        train_data = CarPerspective(
            root=root,
            split='train',
            split_ratio=split_ratio,  # Corrected typo
            transform=train_transform
        )

        test_data = CarPerspective(
            root=root,
            split='test',
            split_ratio=split_ratio,  # Corrected typo
            transform=test_transform
        )

        return train_data, test_data
    else:
        raise AttributeError("Dataset not defined. Please choose a different one.")
