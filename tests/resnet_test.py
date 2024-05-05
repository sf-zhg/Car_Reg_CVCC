import unittest
import torch

from Persp_estimator.models.resnet18 import Resnet18


class TestResnet18Model(unittest.TestCase):
    def setUp(self):
        self.model = Resnet18(feature_dim=5, drop_out_rate=0.5)

    def test_forward(self):
        # Test forward pass with a valid input
        x = torch.rand(8, 3, 8, 8)  # Batch of 8 images, 3 channels, 256x256
        h, g = self.model(x)
        self.assertEqual(h.shape, (8, 1))
        self.assertEqual(g.shape, (8, 1))

    def test_invalid_input(self):
        # Test forward pass with invalid input
        x = torch.rand(3, 8, 8)  # Missing batch dimension
        with self.assertRaises(ValueError):
            self.model(x)
