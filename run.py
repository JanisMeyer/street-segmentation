import os

import click
import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F

from src.config import load_config
from src.data.collect_data import collect_dataset
from src.data.utils import load_index_file
from src.data.dataset import Dataset, calculate_statistics
from src.data.transforms import TransformMask, ToTensor

@click.group()
def cli():
    pass

@cli.command()
@click.argument("data_root")
@click.argument("index_files", nargs=-1)
@click.option("--batch_size", default=20)
@click.option("--num_workers", default=1)
@click.option("--save", is_flag=True, default=False)
@click.option("--config", default="config.json")
def statistics(data_root, index_files, batch_size=20, num_workers=1, save=False, config="config.json"):
    config = load_config(config)
    if not config:
        return
    data_config = config["data"]

    unlabeled = not os.path.exists(os.path.join(data_root, "masks"))

    data = []
    for index_file in index_files:
        data.append(load_index_file(os.path.join(data_root, index_file), unlabeled=unlabeled))
    data = np.concatenate(data, axis=0)
    dataset = Dataset(data_root, data, transforms=[TransformMask(data_config), ToTensor()], unlabeled=unlabeled)
    data_loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)

    calculate_statistics(data_loader, data_config, log_results=True, save_path=os.path.join(data_root, "statistics.pt") if save else None)

@cli.command()
@click.argument("data_root")
@click.argument("output_root")
@click.option("--test_ratio", default=0.2)
@click.option("--val_ratio", default=0.2)
@click.option("--unlabeled", is_flag=True, default=False)
@click.option("--color_mapping", default=None)
@click.option("--images_folder", default="images")
@click.option("--masks_folder", default="masks")
@click.option("--num_workers", default=1)
@click.option("--config", default="config.json")
def collect_data(data_root, output_root, test_ratio=0.2, val_ratio=0.2, unlabeled=False, color_mapping=None, images_folder="images", masks_folder="masks", num_workers=1, config="config.json"):
    config = load_config(config)
    if not config:
        return
    collect_dataset(config, data_root, output_root, test_ratio, val_ratio, unlabeled, color_mapping, images_folder, masks_folder, num_workers=num_workers)

if __name__ == "__main__":
    cli()