from copy import deepcopy

# fixed parameters
workdir = 'workdir'
tag = 'vit_exp'
exp_path = f'{workdir}/{tag}'
project_name = 'vit_classifier'
wandb_path = 'wandb'
checkpoint_path = 'checkpoints'
save_interval = 10
num_classes = 1000
dtype = 'float32'
num_workers = 4

# configurable parameters
seed = 42
lr = 1e-3
epochs = int(1e3)
batch_size = 32
num_warmup_epochs = 10

transform = dict(type='ImageTransform', mode=None)

train_transform = deepcopy(transform)
train_transform['mode'] = 'train'

val_transform = deepcopy(transform)
val_transform['mode'] = 'valid'

test_transform = deepcopy(transform)
test_transform['mode'] = 'test'

dataset = dict(type='ImageDataset', image_dir=None, transform=None)

train_dataset = deepcopy(dataset)
train_dataset.update(
    {'image_dir': 'datasets/imagenet/train', 'transform': train_transform}
)

val_dataset = deepcopy(dataset)
val_dataset.update(
    {'image_dir': 'datasets/imagenet/val', 'transform': val_transform}
)

test_dataset = deepcopy(dataset)
test_dataset.update(
    {'image_dir': 'datasets/imagenet/test', 'transform': test_transform}
)

model = dict(
    type='VisionTransformer',
    image_size=224,
    patch_size=16,
    in_channels=3,
    num_classes=num_classes,
    embed_dim=768,
    depth=12,
    num_heads=12,
    mlp_ratio=4.0,
    dropout=0.1,
)

criterion = dict(type='CrossEntropyLoss', label_smoothing=0.1)

optimizer = dict(type='AdamW', lr=lr, params=None)

scheduler = dict(
    type='CosineWithWarmupScheduler',
    optimizer=None,
    num_warmup_steps=None,
    num_training_steps=None,
)

metrics = [
    dict(type='Accuracy', topk=(1, 5)),
    dict(type='F1Score', average='macro'),
]

trainer = dict(
    type='Trainer',
    model=None,
    train_loader=None,
    valid_loader=None,
    test_loader=None,
    wandb_logger=None,
    accelerator=None,
    optimizer=None,
    scheduler=None,
    criterion=None,
    metrics=None,
    device=None,
    dtype=None,
    exp_path=None,
)
