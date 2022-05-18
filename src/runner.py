import os

import torch

from .models.model import Model
from .logger.reporter import TrainReporter, EvalReporter, InferReporter
from .loss import Loss
from .utils import dict_get
from .logger import log_info, add_log_file

class Runner:
    """ A runner that performs training, evaluation and inference.
    """
    def __init__(self, device, model, optimizer=None, criterion=None):
        self.device = device
        self.model = model

        self.optimizer = optimizer
        self.criterion = criterion

    @staticmethod
    def from_config(config, training=False):
        """ Constructs a runner from the config.
        """
        # setup logging
        log_folder = dict_get(config, "logging", "output_folder")
        if not os.path.exists(log_folder):
            os.makedirs(log_folder)
        add_log_file(os.path.join(log_folder, dict_get(config, "logging", "file_name")))
              
        device = torch.device(dict_get(config, "params", "device", default="cpu"))
        model = Model.from_config(config.get("model")).float().to(device)
        criterion = Loss.from_config(dict_get(config, "params", "losses", default=[]))
        if training:
            optimizer = torch.optim.Adam(model.parameters(),
                                         lr=dict_get(config, "train", "learning_rate", default=0.0),
                                         weight_decay=dict_get(config, "train", "weight_decay", 0.0))
        else:
            optimizer = None

        # load previous checkpoint if continue training or evaluation/inference
        model_path = dict_get(config, "params", "model_path")
        if not training or dict_get(config, "train", "continue", default=False) and os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location="cpu")
            model.load_state_dict(checkpoint["model_state"])
            if optimizer and "optimizer_state" in checkpoint:
                optimizer.load_state_dict(checkpoint["optimizer_state"])
            log_info("Restored model from checkpoint at %s", model_path)
        
        return Runner(device, model, optimizer=optimizer, criterion=criterion)

    def train(self, train_iter, max_steps, reporter, eval_iter=None, report_steps=1000):
        """ Trains the model.
        """
        assert self.optimizer is not None, "Optimizer required for training"
        assert self.criterion is not None, "Criterion required for training"
        assert isinstance(reporter, TrainReporter), "TrainReporter required for training"

        self.model.train()
        reporter.reset()
        for step in range(1, max_steps+1):
            sample = next(train_iter)
            inputs = sample["image"].to(self.device)
            targets = sample["mask"].to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            losses = self.criterion(outputs, targets)
            losses["loss"].backward()
            self.optimizer.step()

            self.reporter.update({
                "step": step,
                "loss": losses["loss"].item(),
                "losses": [loss.item() for loss in losses.get("losses", [])]
            })
            if step % report_steps == 0:
                if eval_iter:
                    self.eval(eval_iter, reporter.eval_reporter, log_results=False)
                    self.model.train()

    def eval(self, eval_iter, reporter, log_results=True):
        """ Evaluates the model.
        """
        self.model.eval()
        reporter.reset()
        for sample in eval_iter:
            inputs = sample["image"].to(self.device)
            targets = sample["mask"].to(self.device)

            with torch.no_grad():
                outputs = self.model(inputs)
                losses = self.criterion(outputs, targets)

            reporter.update({
                "predictions": outputs["predictions"],
                "targets": targets,
                "loss": losses["loss"].item(),
                "losses": [loss.item() for loss in losses.get("losses", [])]
            })
        return reporter.report(log_results=log_results)

    def infer(self, infer_iter, reporter):
        """ Performs inference with the model.
        """
        self.model.eval()
        for sample in infer_iter:
            inputs = sample["image"].to(self.device)
            with torch.no_grad():
                outputs = self.model(inputs)

            reporter.update({
                "predictions": outputs["predictions"]
            })                
        return reporter.report()