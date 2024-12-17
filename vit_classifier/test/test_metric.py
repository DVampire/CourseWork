import unittest

import torch

from src.metric import Accuracy, F1Score


class TestMetrics(unittest.TestCase):
    def setUp(self):
        self.batch_size = 4
        self.num_classes = 3
        self.accuracy = Accuracy(topk=(1, 2))
        self.f1_score = F1Score(average='macro')

        # Create dummy predictions and targets
        self.output = torch.randn(self.batch_size, self.num_classes)
        self.target = torch.randint(0, self.num_classes, (self.batch_size,))

    def test_accuracy(self):
        metrics = self.accuracy(self.output, self.target)
        self.assertIn('Acc@1', metrics)
        self.assertIn('Acc@2', metrics)
        self.assertGreaterEqual(metrics['Acc@1'], 0)
        self.assertLessEqual(metrics['Acc@1'], 100)
        self.assertGreaterEqual(metrics['Acc@2'], metrics['Acc@1'])

    def test_f1_score(self):
        metrics = self.f1_score(self.output, self.target)
        self.assertIn('F1', metrics)
        self.assertGreaterEqual(metrics['F1'], 0)
        self.assertLessEqual(metrics['F1'], 1)


if __name__ == '__main__':
    unittest.main()
