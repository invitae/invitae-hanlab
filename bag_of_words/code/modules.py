import logging

def get_console_logger(name=__name__):
    # Initiate a logger to print messages to the screen.
    # Code reference: https://docs.python.org/2/howto/logging.html#configuring-logging
    # Usage:
    # logger = get_console_logger(name="some_name")
    # logger.info("Some logging information to print to screen.")
    
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    return logger