import torch

from src.registry import METRIC


@METRIC.register_module(force=True)
class Accuracy:
    def __init__(self, topk=(1,)):
        self.topk = topk

    def __call__(self, output, target):
        maxk = max(self.topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, dim=1, largest=True, sorted=True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = {}
        for k in self.topk:
            correct_k = correct[:k].reshape(-1).float().sum(0)
            res[f'Acc@{k}'] = correct_k.mul_(100.0 / batch_size).item()
        return res


@METRIC.register_module(force=True)
class F1Score:
    def __init__(self, average='macro'):
        self.average = average

    def __call__(self, output, target):
        pred = torch.argmax(output, dim=1)
        num_classes = output.size(1)

        tp = torch.zeros(num_classes)
        fp = torch.zeros(num_classes)
        fn = torch.zeros(num_classes)

        for c in range(num_classes):
            tp[c] = ((pred == c) & (target == c)).sum().float()
            fp[c] = ((pred == c) & (target != c)).sum().float()
            fn[c] = ((pred != c) & (target == c)).sum().float()

        epsilon = 1e-7
        precision = tp / (tp + fp + epsilon)
        recall = tp / (tp + fn + epsilon)
        f1 = 2 * (precision * recall) / (precision + recall + epsilon)

        if self.average == 'macro':
            f1 = f1.mean().item()
        elif self.average == 'weighted':
            weights = torch.bincount(target).float() / target.size(0)
            f1 = (f1 * weights).sum().item()
        else:  # micro
            f1 = f1.mean().item()

        return {'F1': f1}
