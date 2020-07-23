# External libs
import os
import json
import logging

def mkdirs(paths):
    """create empty directories if they don't exist

    Parameters:
        paths (str list) -- a list of directory paths
    """
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)

def mkdir(path):
    """create a single empty directory if it didn't exist

    Parameters:
        path (str) -- a single directory path
    """
    if not os.path.exists(path):
        os.makedirs(path)

def load_json(path):
    """Loads a json option file

    Parameters:
        path (str) -- Path to the json file

    Returns:
        Option dictionary
    """
    if (os.path.exists(path)):
        with open(path, 'r') as f:
            config_text = f.read()
        opt = json.loads(config_text)
        logging.info("The config file has been succesfully loaded \n\n")
        return opt
    else:
        error_msg = 'The config file could not be found '
        logging.error(error_msg)
        exit(0)