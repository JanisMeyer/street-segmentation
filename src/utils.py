import os

def get_next_index(folders, index, name_scheme="%0.5d.png"):
    while any(os.path.exists(os.path.join(x, name_scheme % index)) for x in folders):
        index += 1
    return index

def get_architecture_key(model_config):
    architecture_key = model_config.get("architecture", "").lower()
    if "encoder" in model_config:
        architecture_key += "_" + model_config.get("encoder", "").lower()
    if "decoder" in model_config:
        architecture_key += "_" + model_config.get("decoder", "").lower()
    return architecture_key

def dict_get(dictionary, *path, default=None):
    current = dictionary
    for key in path:
        current = current.get(key, default)
        if not isinstance(current, dict):
            return current
    return current


def dict_set(dictionary, value, *path):
    if len(path) == 1:
        dictionary[path[0]] = value
    elif len(path) > 1:
        dictionary[path[0]] = dict_set(dictionary[path[0]], value, *path[1:])
    return dictionary