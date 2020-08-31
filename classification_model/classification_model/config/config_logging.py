import sys
import logging

FORMATTER = logging.Formatter(
"%(asctime)s — %(name)s — %(levelname)s —" "%(funcName)s:%(lineno)d — %(message)s"
)
def config_logger():
    # Streamhandler and sys.stdout allows us to print the log records out in our terminal
    handler = logging.FileHandler("file.log")
    handler.setFormatter(FORMATTER)
    return handler
