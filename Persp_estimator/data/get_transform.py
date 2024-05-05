from torchvision import transforms as t

from typing import Callable


def get_transform(dataset: str, split: str) -> Callable:
    """
    Get data augmentation for images.

    This function returns a callable transformation based on the specified
    dataset and whether the transformation is for training or testing.

    Parameters
    ----------
    dataset : str
        The name of the dataset.
    split : str
        'train' or 'test'.

    Returns
    -------
    Callable
        A callable transformation function.
    """

    if dataset == 'car_persp':
        resize = (256, 256)
        # lets just assume imagenet normalizations as it should fit the
        # dataset quite well. otherwise we would need to write a summary script
        # and calc the mean and std for each channel
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
    else:
        raise AttributeError('dataset not defined choose a different one.')

    if split == 'train':
        # define training transformations
        transform = t.Compose(
            [
                t.Resize(size=resize),
                t.RandomRotation(degrees=(1, 90)),
                t.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                t.RandomHorizontalFlip(p=0.5),
                t.ToTensor(),
                t.Normalize(mean=mean, std=std),
            ]
        )
    elif split == 'test':
        # define test transform
        transform = t.Compose(
            [t.Resize(size=resize), t.ToTensor(), t.Normalize(mean=mean, std=std)]
        )
    else:
        raise AttributeError('split not defined, take test or train')

    return transform
