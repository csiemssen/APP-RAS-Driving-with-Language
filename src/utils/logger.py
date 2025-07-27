import logging
import os
import sys

from src.constants import log_dir

os.makedirs(log_dir, exist_ok=True)

log_file_path = os.path.join(log_dir, "dev.log")

root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)

formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setFormatter(formatter)
root_logger.addHandler(stream_handler)

file_handler = logging.FileHandler(log_file_path)
file_handler.setFormatter(formatter)
root_logger.addHandler(file_handler)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
