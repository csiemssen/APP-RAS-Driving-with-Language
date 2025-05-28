import logging
import sys

# Configure the logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)

def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
