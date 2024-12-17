import logging

from src.registry import LOGGER


@LOGGER.register_module(force=True)
class Logger(logging.Logger):
    def __init__(self, name='logger', level=logging.INFO):
        super().__init__(name, level)

        formatter = logging.Formatter(
            fmt='\u001b[92m%(asctime)s - %(name)s:%(levelname)s\u001b[0m: %(filename)s:%(lineno)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
        )

        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)

        self.addHandler(console_handler)
        self.propagate = False


logger = Logger(name='logger')
