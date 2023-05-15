import os

import yaml

CONFIG_PATH = "../config/"


def load_config(config_name):
    """
    Load a configuration file in YAML format.
   Args:
    config_name (str): The name of the configuration file.

    Returns:
     dict: The loaded configuration as a dictionary.
     """
    with open(os.path.join(CONFIG_PATH, config_name)) as file:
        config = yaml.safe_load(file)

    return config
