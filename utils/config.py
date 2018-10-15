import yaml
from utils.logger import get_console_logger

def load_config(config_fn, config_key):
    logger = get_console_logger(name=__name__)

    config_d = yaml.load(open(config_fn, 'r'))

    logger.info("Config for %s:".format(config_key))
    for k, v in config_d.get(config_key).items():
        logger.info("\t%s: %s" % (k,v))

    return config_d.get(config_key)