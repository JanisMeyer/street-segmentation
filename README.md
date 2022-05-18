# Framework for Semantic Segmentation
(Incomplete) Framework for the training, evaluation and inference of semantic segmentation models (of street images) using PyTorch.

Most of the functionality is exposed through `run.py`. Usage:

- `run.py train` for training
- `run.py eval` for evaluation
- `run.py test` for evaluation on test dataset
- `run.py infer` for inference

The definition of all relevant parameters is given in the config file, by default found at `config.json`.

The framework is incomplete. Most notably, the logging part is under construction.

## Requirements

> Python >= 3.7
> 
> click \
> numpy \
> torch \
> torchvision \
> scikit-image \
> tqdm \
> Pillow