import unittest

import torch
from PIL import Image

from src.transform import ImageTransform


class TestImageTransform(unittest.TestCase):
    def setUp(self):
        self.image_size = 224
        self.train_transform = ImageTransform(mode='train')
        self.val_transform = ImageTransform(mode='valid')
        self.test_transform = ImageTransform(mode='test')

        # Create a dummy RGB image
        self.image = Image.new('RGB', (256, 256))

    def test_train_transform(self):
        transformed = self.train_transform(self.image)
        self.assertIsInstance(transformed, torch.Tensor)
        self.assertEqual(transformed.shape, (3, self.image_size, self.image_size))

    def test_val_transform(self):
        transformed = self.val_transform(self.image)
        self.assertIsInstance(transformed, torch.Tensor)
        self.assertEqual(transformed.shape, (3, self.image_size, self.image_size))

    def test_test_transform(self):
        transformed = self.test_transform(self.image)
        self.assertIsInstance(transformed, torch.Tensor)
        self.assertEqual(transformed.shape, (3, self.image_size, self.image_size))


if __name__ == '__main__':
    unittest.main()
