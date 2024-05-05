import unittest
import os
from Persp_estimator.data.car_persp import CarPerspective

from typing import Any


class TestCarPerspective(unittest.TestCase):
    def setUp(self):
        # setup for tests
        self.root = "raw_data"
        self.split_ratio = 0.7

    def test_initialization(self):
        # test that the dataset initializes correctly
        dataset = CarPerspective(
            root=self.root, split="train", split_ratio=self.split_ratio
        )
        self.assertGreater(len(dataset), 0)

    def test_invalid_split(self):
        # test with an invalid split value
        with self.assertRaises(ValueError):
            CarPerspective(
                root=self.root, split="invalid", split_ratio=self.split_ratio
            )

    def test_missing_data(self):
        # test with a non-existent root directory
        with self.assertRaises(FileNotFoundError):
            CarPerspective(
                root="non_existent", split="train", split_ratio=self.split_ratio
            )

    def test_get_item(self):
        # test fetching an item from the dataset
        dataset = CarPerspective(
            root=self.root, split="train", split_ratio=self.split_ratio
        )
        img, hood_score, backdoor_score = dataset[0]
        self.assertIsInstance(img, Any)
        self.assertIsInstance(hood_score, float)
        self.assertIsInstance(backdoor_score, float)
