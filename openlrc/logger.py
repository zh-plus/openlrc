import logging

from colorlog import ColoredFormatter

handler = logging.StreamHandler()

formatter = ColoredFormatter(
    "%(log_color)s [%(asctime)s] %(levelname)-8s [%(threadName)s] %(message)s",
    datefmt='%Y-%m-%d %H:%M:%S',
    reset=True,
    log_colors={
        'DEBUG': 'cyan',
        'INFO': 'green',
        'WARNING': 'yellow',
        'ERROR': 'red',
        'CRITICAL': 'red,bg_white',
    },
    secondary_log_colors={},
    style='%'
)

handler.setFormatter(formatter)
logger = logging.getLogger()
logger.addHandler(handler)

logger.setLevel('INFO')
