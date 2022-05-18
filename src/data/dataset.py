import os

import torch
import torch.nn.functional as F
from tqdm import tqdm

from .transforms import get_transform, Load
from .utils import load_index_file, get_id_to_label
from ..logger import log_info, log_per_class

def from_config(dataset_configs, index_file, data_config=None, unlabeled=False):
    datasets = [Dataset.from_config(dataset_config, index_file, data_config=data_config, unlabeled=unlabeled) for dataset_config in dataset_configs]
    if len(datasets) == 1:
        return datasets[0]
    else:
        return MultiDataset(*datasets)

class Dataset(torch.utils.data.Dataset):
    """ A dataset with a single datasource.
    """
    def __init__(self, root_folder, data, transforms=None, unlabeled=False):
        self.root_folder = root_folder
        self.data = data
        self.transforms = [Load()]
        if transforms:
            self.transforms += transforms if isinstance(transforms, list) else [transforms]
        self.unlabeled = unlabeled
        log_info("Loaded %s dataset of %d samples from %s." % ("unlabeled" if unlabeled else "labeled", len(self), root_folder))

    @staticmethod
    def from_config(dataset_config, index_file, data_config=None, unlabeled=False):
        data = load_index_file(os.path.join(dataset_config["root"], index_file), unlabeled=unlabeled)
        transforms = [get_transform(transform, data_config=data_config) for transform in dataset_config["transforms"]]
        return Dataset(dataset_config["root"], data, transforms=transforms, unlabeled=unlabeled)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.unlabeled:
            sample = {"image_path": os.path.join(self.root_folder, "images", self.data[idx])}
        else:
            sample = {
                "image_path": os.path.join(self.root_folder, "images", self.data[idx, 0]),
                "mask_path": os.path.join(self.root_folder, "masks", self.data[idx, 1])
            }

        for transform in self.transforms:
            sample = transform(sample)
        return sample

class MultiDataset(torch.utils.data.Dataset):
    """ A dataset with data from multiple sources.
    """
    def __init__(self, *datasets):
        self.datasets = datasets
        self.lengths = [len(dataset) for dataset in self.datasets]
        
    def __len__(self):
        return sum(self.lengths)

    def __getitem__(self, idx):
        dataset_idx = 0
        while idx >= self.lengths[dataset_idx]:
            idx = idx - self.lengths[dataset_idx]
            dataset_idx += 1
        return self.datasets[dataset_idx][idx]

def calculate_statistics(data_loader, data_config, log_results=False, save_path=None):
    """ Calculate and, optionally, store dataset statistics.
    """
    num_classes = len(data_config["labels"])
    id_to_label = get_id_to_label(data_config)
    
    pixel_sum = torch.zeros((3))
    pixel_sq_sum = torch.zeros((3))
    num_pixels = 0
    num_samples = 0
    
    pixels_per_class = torch.zeros((num_classes), dtype=torch.int)
    images_per_class = torch.zeros((num_classes), dtype=torch.int)

    log_info("Calculating dataset statistics...")
    for sample in tqdm(data_loader):
        pixel_sum += torch.sum(sample["image"], dim=[0, 2, 3])
        pixel_sq_sum += torch.sum(sample["image"] ** 2, dim=[0, 2, 3])
        num_pixels += sample["image"].size(0) * sample["image"].size(2) * sample["image"].size(3)
        num_samples += sample["image"].size(0)

        if "mask" in sample:
            target_one_hot = F.one_hot(sample["mask"], num_classes=num_classes)
            pixels_per_class += torch.sum(target_one_hot, dim=[0, 1, 2])
            images_per_class += torch.sum(torch.sum(target_one_hot, dim=[1, 2]) > 0, dim=0)

    mean = pixel_sum / num_pixels
    std = torch.sqrt((pixel_sq_sum / num_pixels) - (mean ** 2))
    
    if log_results:
        log_info("Total number of Samples: %d", num_samples)
        log_info("Total number of Pixels: %d", num_pixels)
        log_info("Per-Pixel Mean: [%0.4f, %0.4f, %0.4f]", mean[0], mean[1], mean[2])
        log_info("Per-Pixel Std: [%0.4f, %0.4f, %0.4f]", std[0], std[1], std[2])
    
    unlabeled = not torch.any(pixels_per_class > 0)
    if not unlabeled and log_results:
        pixels_per_image = num_pixels / num_samples
        per_class_statistics = {
            "num pxs": pixels_per_class,
            r"% of pxs": pixels_per_class / num_pixels * 100,
            "num img": images_per_class,
            r"% of imgs": images_per_class / num_samples * 100,
            "px. / img": pixels_per_class / images_per_class,
            r"% per img": (pixels_per_class / images_per_class) / pixels_per_image * 100
        }
        log_per_class(per_class_statistics, id_to_label, sort_by="num pxs")
    if save_path is not None:
        statistics = {
            "mean": mean, "std": std,
            "num_samples": num_samples,
            "num_pixels": num_pixels,
        }
        if not unlabeled:
            statistics["pixels_per_class"] = pixels_per_class
            statistics["images_per_class"] = images_per_class
        torch.save(statistics, save_path)
        log_info("Written statistics to %s.", save_path)