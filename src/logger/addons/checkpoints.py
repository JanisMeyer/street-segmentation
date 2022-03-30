import torch

from .addon import Addon
from ...utils import dict_get

class CheckpointSaver(Addon):
    def __init__(self, model_path):
        self.model_path = model_path

    @classmethod
    def from_config(cls, config):
        return cls(dict_get(config, "params", "model_path"))

    @property
    def required_update_fields():
        return ["model"]

    def update(self, update_state):
        self.save_checkpoint(
            update_state["model"].state_dict(),
            update_state["optimizer"].state_dict() if "optimizer" in update_state else None,
            update_state["logger"].state_dict() if "logger" in update_state else None,
            update_state["lr_scheduler"].state_dict() if "lr_scheduler" in update_state else None
        )

    def save_checkpoint(self, model_state, optimizer_state=None, logger_state=None, lr_state=None):
        state_dict = {"model_state": model_state}
        if optimizer_state:
            state_dict["optimizer_state"] = optimizer_state
        if logger_state:
            state_dict["logger_state"] = logger_state
        if lr_state:
            state_dict["lr_state"] = lr_state
        torch.save(state_dict, self.model_path)

    def report(self):
        return {
            "stored_checkpoint": torch.load(self.model_path),
            "repr": "Stored latest checkpoint at %s" % self.model_path
        }

class BestCheckpointSaver(CheckpointSaver):
    def __init__(self, model_path, ):
        super(BestCheckpointSaver, self).__init__(model_path)
        