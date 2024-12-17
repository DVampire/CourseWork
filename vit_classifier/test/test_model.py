import unittest

import torch

from src.model import VisionTransformer


class TestVisionTransformer(unittest.TestCase):
    def setUp(self):
        self.batch_size = 4
        self.num_classes = 10
        self.model = VisionTransformer(
            image_size=224,
            patch_size=16,
            in_channels=3,
            num_classes=self.num_classes,
            embed_dim=768,
            depth=12,
            num_heads=12,
            mlp_ratio=4.0,
            dropout=0.1,
        )

    def test_forward(self):
        x = torch.randn(self.batch_size, 3, 224, 224)
        output = self.model(x)
        self.assertEqual(output.shape, (self.batch_size, self.num_classes))

    def test_patch_embedding(self):
        x = torch.randn(self.batch_size, 3, 224, 224)
        patches = self.model.patch_embed(x)
        num_patches = (224 // 16) ** 2
        self.assertEqual(patches.shape, (self.batch_size, num_patches, 768))


if __name__ == '__main__':
    unittest.main()
