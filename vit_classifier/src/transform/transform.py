from torchvision import transforms

from src.logger import logger
from src.registry import TRANSFORM


@TRANSFORM.register_module(force=True)
class ImageTransform(transforms.Compose):
    def __init__(self, mode='train'):
        if mode == 'train':
            logger.info('Using training transforms.')
            super().__init__(
                [
                    transforms.RandomResizedCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ColorJitter(0.4, 0.4, 0.4),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]
                    ),
                ]
            )
        elif mode == 'valid':
            logger.info('Using validation transforms.')
            super().__init__(
                [
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]
                    ),
                ]
            )
        elif mode == 'test':
            logger.info('Using test transforms.')
            super().__init__(
                [
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]
                    ),
                ]
            )
        else:
            raise ValueError(
                f"Unsupported mode: {mode}. Use 'train', 'valid', or 'test'."
            )
