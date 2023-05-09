import os

import yaml

CONFIG_PATH = "../config/"


def load_config(config_name):
    """

    Args:
        config_name:

    Returns:

    """
    with open(os.path.join(CONFIG_PATH, config_name)) as file:
        config = yaml.safe_load(file)

    return config
