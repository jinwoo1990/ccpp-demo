import logging.config
import json
import os


LOG_CONFIG_PATH = './config/loggers_config.json'

logging.config.dictConfig(json.load(open(LOG_CONFIG_PATH)))
logger = logging.getLogger('base_logger')
