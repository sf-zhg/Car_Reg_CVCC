from torchvision import transforms as t

from typing import Callable


def get_transform(dataset: str, split: str) -> Callable:
    """
    function to get data augmentation for images. for simplicity sake, i do not
    define more than two input args but to be precise, we would have to define
    transformation modules as args as well as the transformation params.

    Parameters
    ----------
    dataset : str
        dataset name as defined in the code.
    split : str
        train or test.

    Returns
    -------
    Callable
        transformation functions.

    """

    if dataset == "car_persp":
        resize = (256, 256)
        # lets just assume imagenet normalizations as it should fit the
        # dataset quite well. otherwise we would need to write a summary script
        # and calc the mean and std for each channel
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
    else:
        raise AttributeError(f"dataset not defined choose a different one.")

    if split == "train":
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
    elif split == "test":
        # define test transform
        transform = t.Compose(
            [t.Resize(size=resize), t.ToTensor(), t.Normalize(mean=mean, std=std)]
        )
    else:
        raise AttributeError(f"split not defined, take test or train")

    return transform
