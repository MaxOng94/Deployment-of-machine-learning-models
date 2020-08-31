import logging
from classification_model.config.config import PACKAGE_ROOT
from classification_model.config.config_logging import config_logger
import sys
# We want to create our logger here
# to get the absolute version path
VERSION_PATH = PACKAGE_ROOT/'VERSION'

logger= logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

handler= config_logger()
logger.addHandler(handler)

with open(VERSION_PATH,"r") as version_file:
    __version__ = version_file.read().strip()
