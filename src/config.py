import os
import json

from .data.transforms import TRANSFORMS
from .logger import log_info

def load_config(path):
    """ Loads and verifies the json-config at the given path.
    """
    if not os.path.exists(path):
        log_info("Config file at %s could not be found.", path)
        return None 
    with open(path, "r") as f:
        config = json.loads(f.read())
    if verify_config(config):
        log_info("Parsed config file...")
        return config
    else:
        log_info("Invalid config file")
        return None

def verify_config(config):
    if not isinstance(config, dict):
        return False
    valid_keys = {
        "params": verify_params_config,
        "train": verify_train_config,
        "data": verify_data_config
    }
    for key in config:
        if key not in valid_keys or not valid_keys[key](config[key]):
            log_info("Error when parsing config: %s", key)
            return False
    return True 

def verify_params_config(config):
    if not isinstance(config, dict):
        return False
    valid_keys = {
        
    }
    for key in config:
        if key not in valid_keys or not valid_keys[key](config[key]):
            return False
    return True

def verify_train_config(config):
    if not isinstance(config, dict):
        return False
    valid_keys = {
        
    }
    for key in config:
        if key not in valid_keys or not valid_keys[key](config[key]):
            return False
    return True

def verify_data_config(config):
    if not isinstance(config, dict):
        return False
    valid_keys = {
        "height": is_int,
        "width": is_int,
        "preprocess": (lambda x: is_list(x, is_transform)),
        "labeled": (lambda x: is_list(x, is_dataset)),
        "unlabeled": (lambda x: is_list(x, is_dataset)),
        "labels": verify_labels_config
    }
    for key in config:
        if key not in valid_keys or not valid_keys[key](config[key]):
            return False
    return True

def verify_labels_config(config):
    if not isinstance(config, list):
        return False
    valid_keys = {
        "id": is_id,
        "label": is_any,
        "color": is_color
    }
    ids = []
    for element in config:
        if not isinstance(element, dict):
            return False
        for key in element:
            if key not in valid_keys or not valid_keys[key](element[key]):
                return False
        for key in valid_keys:
            if key not in element:
                return False
        if element["id"] in ids:
            return False
        ids.append(element["id"])
    if max(ids) != (len(ids) - 1):
        return False
    return True

def is_dataset(entry):
    valid_keys = {
        "root": is_root,
        "transforms": (lambda x: is_list(x, is_transform))
    }
    for key in entry:
        if key not in valid_keys or not valid_keys[key](entry[key]):
            return False
    return True
   
def is_int(entry):
    return isinstance(entry, int)

def is_id(entry):
    return isinstance(entry, int) and entry >= 0

def is_root(entry):
    if not os.path.isdir(entry):
        return False
    if not os.path.isdir(os.path.join(entry, "images")):
        return False
    return True

def is_any(entry):
    return True

def is_color(entry):
    if not isinstance(entry, list):
        return False
    for value in entry:
        if not is_int(value) or value < 0 or value > 255:
            return False
    return True

def is_transform(entry):
    return entry in TRANSFORMS

def is_list(entry, predicate=None):
    if not isinstance(entry, list):
        return False
    for element in entry:
        if not predicate(element):
            return False
    return True