__version__ = '0.9.4'

import sys

try:
    from loguru import logger
    # Remove default handler
    logger.remove()
    # Add custom handler with clean format including module and line number
    logger.add(
        sys.stderr,
        format="<green>{time:HH:mm:ss}</green> | <cyan>{module:>16}:{line}</cyan> | <level>{level: >8}</level> | <level>{message}</level>",
        colorize=True,
        level="INFO" # "DEBUG" to enable logger.debug("message") and up prints 
                     # "ERROR" to enable only logger.error("message") prints
                     # etc
    )
    # Disable before release or as needed
    logger.disable("kokoro")
except ImportError:
    import logging
    logger = logging.getLogger("kokoro")

try:
    from .model import KModel
    from .pipeline import KPipeline
except ImportError:
    pass

from .hf_utils import (
    get_checkpoints_dir,
    set_checkpoints_dir,
    resolve_model_path,
    resolve_pretrained_path,
    load_custom_model,
)


