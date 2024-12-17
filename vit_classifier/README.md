# Vision Transformer (ViT) Image Classifier

This project implements a Vision Transformer (ViT) model for image classification tasks. The implementation is based on the paper ["An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale"](https://arxiv.org/abs/2010.11929).

## Features

- Vision Transformer (ViT) implementation with configurable architecture
- Support for distributed training using Accelerate
- Wandb integration for experiment tracking
- Modular and extensible design using MMEngine
- Comprehensive metrics including accuracy and F1-score
- Configurable data augmentation pipeline
- Learning rate scheduling with warmup

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/vit_classifier.git
cd vit_classifier
```

2. Install the required packages:
```bash
pip install -r requirements.txt
```

## Project Structure

```
vit_classifier/
├── src/                         # Main package containing core functionality
│   ├── config/                  # Configuration management
│   ├── criterion/              # Loss functions
│   ├── dataset/               # Dataset handling
│   ├── logger/                # Logging utilities
│   ├── metric/                # Evaluation metrics
│   ├── model/                 # Model architectures
│   ├── optimizer/             # Optimizers
│   ├── scheduler/             # Learning rate schedulers
│   ├── trainer/               # Training logic
│   ├── transform/             # Data transformations
│   ├── utils/                 # Utility functions
│   ├── __init__.py
│   └── registry.py            # Component registry
├── configs/                    # Configuration files
├── datasets/                   # Dataset directory
├── test/                      # Unit tests
├── run.py                     # Main entry point
├── README.md
└── requirements.txt
```

## Usage

1. Prepare your dataset in the following structure:
```
datasets/
└── your_dataset/
    ├── train/
    │   ├── class1/
    │   │   ├── img1.jpg
    │   │   └── ...
    │   └── class2/
    │       ├── img1.jpg
    │       └── ...
    ├── val/
    └── test/
```

2. Configure your experiment in `configs/vit.py`:
- Adjust model architecture parameters
- Set training hyperparameters
- Configure data augmentation
- Set up logging and checkpointing

3. Run training:
```bash
python run.py --config configs/vit.py --tag experiment_name
```

Additional arguments:
- `--workdir`: Working directory for saving experiments
- `--seed`: Random seed for reproducibility
- `--device`: Device to use (cuda/cpu)
- `--if_remove`: Remove existing experiment directory

## Model Architecture

The Vision Transformer (ViT) architecture consists of:
- Patch embedding layer
- Position embeddings
- Transformer encoder blocks with multi-head self-attention
- MLP head for classification

Key configurable parameters:
- `image_size`: Input image size
- `patch_size`: Size of image patches
- `embed_dim`: Embedding dimension
- `depth`: Number of transformer blocks
- `num_heads`: Number of attention heads
- `mlp_ratio`: MLP hidden dimension ratio

## Training

The training process includes:
1. Data augmentation with random resizing, cropping, and flipping
2. Learning rate warmup and cosine decay
3. Label smoothing for better generalization
4. Distributed training support
5. Progress tracking with Wandb
6. Regular model checkpointing

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this code in your research, please cite:

```bibtex
@article{dosovitskiy2020image,
  title={An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale},
  author={Dosovitskiy, Alexey and Beyer, Lucas and Kolesnikov, Alexander and Weissenborn, Dirk and Zhai, Xiaohua and Unterthiner, Thomas and Dehghani, Mostafa and Minderer, Matthias and Heigold, Georg and Gelly, Sylvain and others},
  journal={arXiv preprint arXiv:2010.11929},
  year={2020}
}
```
