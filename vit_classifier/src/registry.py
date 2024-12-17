from mmengine.registry import Registry

DATASET = Registry('dataset', locations=['src.dataset'])
TRANSFORM = Registry('transform', locations=['src.transform'])

MODEL = Registry('model', locations=['src.model'])
OPTIMIZER = Registry('optimizer', locations=['src.optimizer'])
SCHEDULER = Registry('scheduler', locations=['src.scheduler'])
CRITERION = Registry('criterion', locations=['src.criterion'])

LOGGER = Registry('logger', locations=['src.logger'])

METRIC = Registry('metric', locations=['src.metric'])
TRAINER = Registry('trainer', locations=['src.trainer'])

CONFIG = Registry('config', locations=['src.config'])
