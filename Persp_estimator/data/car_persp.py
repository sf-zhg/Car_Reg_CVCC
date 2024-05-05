import os
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image
import random

from typing import Any, Callable, Optional, Tuple


class CarPerspective(Dataset):
    """
    class for loading the dataset
    """

    def __init__(
        self,
        root: str,
        split: str,
        split_ratio: float,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ) -> None:
        """
        initializes the dataset class with:

        root: root directory containing the image and csv files, argument has
        to be a string.

        split: whether to create the training set or the test set, string
        either 'train' or 'test'.

        transform: optional transformations to apply to the images as a
        function. obviously transformations for the test set should only
        contain resize and scalling.

        split_ratio: ratio to split the dataset into train and test sets for
        simplicity sake. in reality it would be wise to xarefully construct
        the sets and not randomly to prevent  long-tailed distributions and
        sufficient variety during training.
        """
        # Set the paths for images and the csv files
        full_root = os.path.join(root, "CodingChallenge_v2")
        img_path = os.path.join(full_root, "imgs")
        csv_path = os.path.join(full_root, "car_imgs_4000.csv")

        if not os.path.exists(img_path):
            raise FileNotFoundError(f"image folder '{img_path}' does not exist.")

        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"csv file '{csv_path}' does not exist.")

        # read csv file
        self.annotations = pd.read_csv(csv_path)

        # store image path and transformations for later use
        self.img_path = img_path
        self.transform = transform
        self.target_transform = target_transform

        # store split string and limit argument inputs as a failsafe
        self.split = split
        assert self.split == "train" or self.split == "test"

        # randomly split the dataset into train and test sets
        random.seed(42)
        dataset_length = len(self.annotations)
        train_length = int(split_ratio * dataset_length)

        # shuffle the data before splitting
        indices = list(range(dataset_length))
        random.shuffle(indices)

        # get indices for train and test set
        train_indices = indices[:train_length]
        test_indices = indices[train_length:]

        # reset the indices such that both train and test start form the beginning
        if self.split == "train":
            self.annotations = self.annotations.iloc[train_indices].reset_index(
                drop=True
            )
        elif self.split == "test":
            self.annotations = self.annotations.iloc[test_indices].reset_index(
                drop=True
            )
        else:
            raise AttributeError(f"argument not in train, test.")

    def __len__(self) -> int:
        """
        returns the total number of samples in the dataset.
        """
        return len(self.annotations)

    def __getitem__(self, index: int) -> Tuple[Any, float, float]:
        """
        fetches the image and annotation scores based on the given index.

        index: index of the item to fetch.
        """
        # get file name from the annotations, we can also use columns index to
        # access filename as it is in the first column
        img_name = self.annotations.iloc[index]["filename"]
        img_file_path = os.path.join(self.img_path, img_name)

        # load the image
        image = Image.open(img_file_path)

        # apply any transformation to umage
        if self.transform:
            image = self.transform(image)

        # get the perspective scores from the csv file
        hood_score = self.annotations.iloc[index]["perspective_score_hood"]
        backdoor_score = self.annotations.iloc[index]["perspective_score_backdoor_left"]

        # apply target transform
        if self.target_transform:
            hood_score = self.target_transform(hood_score)
            backdoor_score = self.target_transform(backdoor_score)

        return image, hood_score, backdoor_score
