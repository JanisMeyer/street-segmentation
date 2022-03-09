import os
from random import shuffle
import json
from threading import Thread

import numpy as np
from PIL.ImageColor import getcolor

from ..logger import log_info
from .transforms import get_transform, Load
from ..utils import get_next_index
from .utils import list_folders, get_samples, save_image

def collect_dataset(config, data_root, output_root, test_ratio=0.2, val_ratio=0.2, unlabeled=False, color_mapping=None, images_folder="images", masks_folder="masks", num_workers=1):
    series = []
    for folder in list_folders(data_root):
        if not os.path.exists(os.path.join(data_root, folder, images_folder)):
            continue
        if not os.path.exists(os.path.join(data_root, folder, masks_folder)) and not unlabeled:
           continue
        samples = get_samples(os.path.join(data_root, folder), images_folder, masks_folder, unlabeled)
        if len(samples) > 0:
            series.append((os.path.join(data_root, folder), len(samples), samples))
    
    num_samples = sum(x[1] for x in series)
    log_info("Collected %d samples...", num_samples)
    if unlabeled:
        train_data = series
        test_data = []
        val_data = []
    else:
        num_test = test_ratio * num_samples
        num_val = val_ratio * (num_samples - num_test)
        
        shuffle(series)
        test_data = []
        val_data = []
        train_data = []
        for data in series:
            if num_test > 0:
                test_data += data[2]
                num_test = num_test - data[1]
            elif num_val > 0:
                val_data += data[2]
                num_val = num_val - data[1]
            else:
                train_data += data[2]
    log_info("Number of samples in train: %d", len(train_data))
    log_info("Number of samples in val: %d", len(val_data))
    log_info("Number of samples in test: %d", len(test_data))
    
    if not os.path.exists(output_root):
        os.makedirs(output_root)
    if not os.path.exists(os.path.join(output_root, "images")):
        os.makedirs(os.path.join(output_root, "images"))
    if not os.path.exists(os.path.join(output_root, "masks")) and not unlabeled:
        os.makedirs(os.path.join(output_root, "masks"))

    data_config = config["data"]
    transforms = [Load()]
    for transform in data_config.get("preprocess", []):
        transforms.append(get_transform(transform, data_config=data_config))
    
    if color_mapping:
        with open(color_mapping, "r") as f:
            mapping = json.loads(f.read())
        
        mapping = {getcolor(key, "RGB"): getcolor(value, "RGB") for key, value in mapping.items()}
        
        def map(arr):
            for key, value in mapping.items():
                arr[np.all(arr == key, axis=-1)] = value
            return arr
        color_map = map
    else:
        color_map = None
    
    index = get_next_index([os.path.join(output_root, "images"), os.path.join(output_root, "masks")], 0)
    samples = []
    
    log_info("Writing index files...")
    with open(os.path.join(output_root, "test.txt"), "a") as f:
        for sample in test_data:
            name = "%0.5d.png" % index
            f.write("%s %s\n" % (name, name))
            samples.append((name, sample))
            index += 1
    with open(os.path.join(output_root, "val.txt"), "a") as f:
        for sample in val_data:
            name = "%0.5d.png" % index
            f.write("%s %s\n" % (name, name))
            samples.append((name, sample))
            index += 1
    with open(os.path.join(output_root, "train.txt"), "a") as f:
        for sample in train_data:
            name = "%0.5d.png" % index
            if unlabeled:
                f.write("%s\n" % (name))
            else:
                f.write("%s %s\n" % (name, name))
            samples.append((name, sample))
            index += 1
    
    log_info("Writing images...")
    if num_workers > 1:
        num_samples_per_worker = len(samples) // num_workers
        threads = []
        for idx in range(num_workers):
            if id == num_workers - 1:
                _samples = samples[idx * num_samples_per_worker:]
            else:
                _samples = samples[idx * num_samples_per_worker : (idx + 1) * num_samples_per_worker]
            thread = Thread(target=save_samples,
                            args=(output_root, _samples, transforms, color_map))
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()
    else:
        save_samples(output_root, samples, transforms, color_map=None)
    log_info("Done.")

def save_samples(root_folder, samples, transforms, color_map=None):
    for name, sample in samples:
        for transform in transforms:
            sample = transform(sample) 
        if color_map is not None:
            sample["mask"] = color_map(sample["mask"])
        save_image(os.path.join(root_folder, "images", name), sample["image"])
        if "mask" in sample:
            save_image(os.path.join(root_folder, "masks", name), sample["mask"])
