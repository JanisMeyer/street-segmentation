import os

import click
import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F

from src.config import load_config
from src.data.collect_data import collect_dataset
from src.data.utils import load_index_file
from src.data.dataset import Dataset, MultiDataset, calculate_statistics
from src.data.transforms import TransformMask, ToTensor
from src.runner import Runner

@click.group()
def cli():
    pass

@cli.command()
@click.option("--config", default="config.json")
def train(config="config.json"):
    """ Trains the model as defined in the provided config file. Uses "config.json" per default.
    """
    config = load_config(config)
    if not config:
        return
    data_config = config["data"]
    train_config = config["train"]
    eval_config = config["eval"]
    
    dataset = MultiDataset(*[Dataset.from_config(dataset_config, train_config["index_file"], data_config) for dataset_config in data_config["labeled"]])
    data_loader = DataLoader(dataset, batch_size=train_config["batch_size"], shuffle=True)

    eval_dataset = MultiDataset(*[Dataset.from_config(dataset_config, eval_config["index_file"], data_config) for dataset_config in data_config["labeled"]])
    eval_iter = DataLoader(eval_dataset, batch_size=eval_config["batch_size"])
    
    runner = Runner.from_config(config, training=True)
    runner.train(data_loader, train_config["max_steps"], reporter, eval_iter=eval_iter, report_steps=train_config["report_steps"])

@cli.command()
@click.option("--config", default="config.json")
def eval(config="config.json"):
    """ Evaluates the model on the validation dataset as defined in the provided config file. Uses "config.json" per default.
    """
    config = load_config(config)
    if not config:
        return
    data_config = config["data"]
    eval_config = config["eval"]

    eval_dataset = MultiDataset(*[Dataset.from_config(dataset_config, eval_config["index_file"], data_config) for dataset_config in data_config["labeled"]])
    eval_iter = DataLoader(eval_dataset, batch_size=eval_config["batch_size"])
    
    runner = Runner.from_config(config)
    runner.eval(eval_iter, reporter)

@cli.command()
@click.option("--config", default="config.json")
def test(config="config.json"):
    """ Evaluates the model on the test dataset as defined in the provided config file. Uses "config.json" per default.
    """
    config = load_config(config)
    if not config:
        return
    data_config = config["data"]
    test_config = config["test"]
    
    test_dataset = MultiDataset(*[Dataset.from_config(dataset_config, test_config["index_file"], data_config) for dataset_config in data_config["labeled"]])
    test_iter = DataLoader(test_dataset, batch_size=test_config["batch_size"])
    
    runner = Runner.from_config(config)
    runner.eval(test_iter, reporter)

@cli.command()
@click.option("--config", default="config.json")
def infer(config="config.json"):
    """ Performs inference with the model as defined in the provided config file using the inference data defined in the config file. Uses "config.json" per default.
    """
    config = load_config(config)
    if not config:
        return
    data_config = config["data"]
    infer_config = config["infer"]

    infer_dataset = MultiDataset(*[Dataset.from_config(dataset_config, infer_config["index_file"], data_config) for dataset_config in data_config["labeled"]])
    infer_iter = DataLoader(infer_dataset, batch_size=infer_config["batch_size"])
    
    runner = Runner.from_config(config)
    report = runner.infer(infer_iter, reporter)

@cli.command()
@click.argument("data_root")
@click.argument("index_files", nargs=-1)
@click.option("--batch_size", default=20)
@click.option("--num_workers", default=1)
@click.option("--save", is_flag=True, default=False)
@click.option("--config", default="config.json")
def statistics(data_root, index_files, batch_size=20, num_workers=1, save=False, config="config.json"):
    """Calculates and stores dataset statistics.

    Args:
        data_root (str): The path to the dataset root.
        index_files ([str]): List of index files relative to the provided root.
        batch_size (int, optional): The batch size to use when handling the data. Defaults to 20.
        num_workers (int, optional): Number of parallel workers for data handling. Defaults to 1.
        save (bool, optional): Whether to store the statistics in "statistics.pt" in the dataset root. Defaults to False.
        config (str, optional): Path to the config file. Defaults to "config.json".
    """
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
    """ Collects data and builds a dataset.

    Args:
        data_root (str): Path to the data to collect.
        output_root (str): Path to the root folder for storing the new dataset. 
        test_ratio (float, optional): Ratio of samples to use for the test dataset. Defaults to 0.2.
        val_ratio (float, optional): Ratio of remaining samples to use for the validation dataset. Defaults to 0.2.
        unlabeled (bool, optional): Whether the samples have no labels. Defaults to False.
        color_mapping (str, optional): Path to a file defining a color mapping for combining classes in the labels. Defaults to None.
        images_folder (str, optional): Name of the folder to find the images in. Defaults to "images".
        masks_folder (str, optional): Name of the folder to find the labels in. Defaults to "masks".
        num_workers (int, optional): Number of parallel workers. Defaults to 1.
        config (str, optional): Path to the config file. Defaults to "config.json".
    """
    config = load_config(config)
    if not config:
        return

    collect_dataset(config, data_root, output_root, test_ratio, val_ratio, unlabeled, color_mapping, images_folder, masks_folder, num_workers=num_workers)
if __name__ == "__main__":
    cli()