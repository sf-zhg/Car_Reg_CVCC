import os
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image
import random

from typing import Any, Callable, Optional, Tuple


class CarPerspective(Dataset):
    '''
    Class for loading the Car Perspective dataset.
    '''
    def __init__(
        self,
        root: str,
        split: str,
        split_ratio: float,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ) -> None:
        """
        Initializes the dataset with:
        - root: Root directory containing images and CSV files.
        - split: Whether to create the training or test set ('train' or 'test').
        - transform: Optional transformations to apply to images.
        - split_ratio: Ratio to split the dataset into training and testing.
        """
        # Paths to image and CSV data
        full_root = os.path.join(root, 'CodingChallenge_v2')
        img_path = os.path.join(full_root, 'imgs')
        csv_path = os.path.join(full_root, 'car_imgs_4000.csv')

        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image folder '{img_path}' does not exist.")

        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV file '{csv_path}' does not exist.")

        # Read CSV data
        self.annotations = pd.read_csv(csv_path)

        # Store image path and transformations for later use
        self.img_path = img_path
        self.transform = transform
        self.target_transform = target_transform

        # Validate the split argument
        if split not in ('train', 'test'):
            raise ValueError("Split must be either 'train' or 'test'.")

        self.split = split

        # Randomly split the dataset into train and test sets
        random.seed(42)  # Ensure reproducibility
        dataset_length = len(self.annotations)
        train_length = int(split_ratio * dataset_length)

        # Shuffle the indices before splitting
        indices = list(range(dataset_length))
        random.shuffle(indices)

        # Get indices for train and test sets
        train_indices = indices[:train_length]
        test_indices = indices[train_length:]

        # Reset indices for train and test sets
        if self.split == 'train':
            self.annotations = self.annotations.iloc[train_indices].reset_index(drop=True)
        elif self.split == 'test':
            self.annotations = self.annotations.iloc[test_indices].reset_index(drop=True)

    def __len__(self) -> int:
        """
        Returns the total number of samples in the dataset.
        """
        return len(self.annotations)

    def __getitem__(self, index: int) -> Tuple[Any, float, float]:
        """
        Fetches the image and annotation scores based on the given index.
        """
        # Get the filename from the annotations
        img_name = self.annotations.iloc[index]['filename']
        img_file_path = os.path.join(self.img_path, img_name)

        # Load the image
        image = Image.open(img_file_path)

        # Apply transformations to the image
        if self.transform:
            image = self.transform(image)

        # Get the perspective scores from the CSV
        hood_score = self.annotations.iloc[index]['perspective_score_hood']
        backdoor_score = self.annotations.iloc[index]['perspective_score_backdoor_left']

        # Apply target transformations if specified
        if self.target_transform:
            hood_score = self.target_transform(hood_score)
            backdoor_score = self.target_transform(backdoor_score)

        return image, hood_score, backdoor_score
