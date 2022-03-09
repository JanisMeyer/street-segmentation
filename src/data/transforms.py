import sys
import inspect

import torch
import numpy as np

from .utils import load_image, resize_image, resize_mask, get_color_map

def get_transform(class_name, *args, **kwargs):
    if class_name in TRANSFORMS:
        return TRANSFORMS[class_name](*args, **kwargs)

class Transform:
    pass

class Load(Transform):
    def __call__(self, sample):
        if "image_path" in sample:
            sample["image"] = load_image(sample["image_path"])
        if "mask_path" in sample:
            sample["mask"] = load_image(sample["mask_path"])
        return sample

class ToTensor(Transform):
    def __call__(self, sample):
        if "image" in sample:
            sample["image"] = torch.from_numpy(np.transpose(sample["image"], (2, 0, 1)) / 255.0).float()
        if "mask" in sample:
            sample["mask"] = torch.from_numpy(sample["mask"]).long()
        return sample

class TransformMask(Transform):
    def __init__(self, data_config):
        self.color_map = get_color_map(data_config)

    def __call__(self, sample):
        if "mask" in sample:
            mask = np.dot(sample["mask"], [1, 256, 256**2])
            sample["mask"] = self.color_map[mask]
        return sample

class Resize(Transform):
    def __init__(self, data_config):
        self.width = data_config["width"]
        self.height = data_config["height"]

    def __call__(self, sample):
        if "image" in sample:
            sample["image"] = resize_image(sample["image"], (self.height, self.width))
        if "mask" in sample:
            sample["mask"] = resize_mask(sample["mask"], (self.height, self.width))
        return sample

def transform_predicate(member):
    return inspect.isclass(member) and issubclass(member, Transform)

TRANSFORMS = {}
for class_name, cls in inspect.getmembers(sys.modules[__name__], transform_predicate):
    TRANSFORMS[class_name] = cls