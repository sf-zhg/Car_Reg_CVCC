import unittest
from torchvision import transforms
from Persp_estimator.data.get_transform import get_transform


class TestGetTransform(unittest.TestCase):
    def test_train_transform(self):
        # test the training transformation
        transform = get_transform(dataset="car_persp", split="train")
        self.assertIsInstance(transform, transforms.Compose)

    def test_test_transform(self):
        # test the test transformation
        transform = get_transform(dataset="car_persp", split="test")
        self.assertIsInstance(transform, transforms.Compose)

    def test_invalid_dataset(self):
        # test with an invalid dataset name
        with self.assertRaises(AttributeError):
            get_transform(dataset="invalid_dataset", split="train")

    def test_invalid_split(self):
        # test with an invalid split value
        with self.assertRaises(AttributeError):
            get_transform(dataset="car_persp", split="invalid_split")
