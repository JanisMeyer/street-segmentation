import torch

from ...utils import dict_get
from .addon import Addon

class Metric(Addon):
    def __init__(self):
        self.num_updates = 0

    def reset(self):
        self.num_updates = 0

    def update(self, update_state):
        self.num_updates += 1

    def report(self):
        return {
            "num_updates": self.num_updates
        }

class Loss(Metric):
    def __init__(self):
        super(Loss, self).__init__()
        self.loss = 0.0
        self.losses = {}

    @property
    def required_update_fields():
        return ["loss"]

    def reset(self):
        super(Loss, self).reset()
        self.loss = 0.0
        self.losses = {}

    def update(self, update_state):
        super(Loss, self).update(update_state)
        self.loss += update_state["loss"]
        if "losses" in update_state:
            for loss_name in update_state["losses"]:
                if loss_name not in self.losses:
                    self.losses[loss_name] = dict_get(update_state, "losses", loss_name)
                else:
                    self.losses[loss_name] += dict_get(update_state, "losses", loss_name)

    def report(self):
        if len(self.losses) > 1:
            repr = "(%s)" % " + ".join(["%0.4f" % (self.losses[key] / self.num_updates) for key in self.losses])
        else:
            repr = ""
        loss = self.loss / self.num_updates
        return {
            "loss": loss,
            "repr": "Loss: %0.4f%s" % (loss, repr)
        }

class Accuracy(Metric):
    def __init__(self, num_classes):
        super(Accuracy, self).__init__()
        self.num_classes = num_classes

        self.num_correct = torch.zeros(self.num_classes)
        self.num_pixels = torch.zeros(self.num_classes)

    @classmethod
    def from_config(cls, config):
        return cls(len(dict_get(config, "data", "labels", default=[])))

    @property
    def required_update_fields():
        return ["intersection", "num_relevant"]

    def reset(self):
        super(Accuracy, self).reset()
        self.num_correct = torch.zeros(self.num_classes)
        self.num_pixels = torch.zeros(self.num_classes)

    def update(self, update_state):
        super(Accuracy, self).update(update_state)
        self.num_correct += update_state["intersection"]
        self.num_pixels += update_state["num_relevant"]

    def report(self):
        per_class_score = self.num_correct / self.num_pixels
        accuracy = torch.sum(self.num_correct) / torch.sum(self.num_pixels)
        return {
            "accuracy": accuracy,
            "per_class_score": per_class_score,
            "repr": "Accuracy: %0.4f" % accuracy
        }

class Predicted(Metric):
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.num_predicted = torch.zeros(self.num_classes, dtype=torch.int)

    @classmethod
    def from_config(cls, config):
        return cls(len(dict_get(config, "data", "labels", default=[])))

    @property
    def required_update_fields():
        return ["num_predicted"]

    def reset(self):
        self.num_predicted = torch.zeros(self.num_classes, dtype=torch.int)

    def update(self, update_state):
        self.num_predicted += update_state["num_predicted"]

    def report(self):
        return {
            "num_predicted": torch.sum(self.num_predicted),
            "per_class_score": self.num_predicted,
            "repr": "Predicted: %d" % torch.sum(self.num_predicted)
        }

class Relevant(Metric):
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.num_relevant = torch.zeros(self.num_classes, dtype=torch.int)

    @classmethod
    def from_config(cls, config):
        return cls(len(dict_get(config, "data", "labels", default=[])))

    @property
    def required_update_fields():
        return ["num_relevant"]

    def reset(self):
        self.num_relevant = torch.zeros(self.num_classes, dtype=torch.int)

    def update(self, update_state):
        self.num_relevant += update_state["num_relevant"]

    def report(self):
        return {
            "num_relevant": torch.sum(self.num_relevant),
            "per_class_score": self.num_relevant,
            "repr": "Relevant: %d" % torch.sum(self.num_relevant)
        }

class Correct(Metric):
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.num_correct = torch.zeros(self.num_classes, dtype=torch.int)

    @classmethod
    def from_config(cls, config):
        return cls(len(dict_get(config, "data", "labels", default=[])))

    @property
    def required_update_fields():
        return ["num_correct"]

    def reset(self):
        self.num_correct = torch.zeros(self.num_classes, dtype=torch.int)

    def update(self, update_state):
        self.num_correct += update_state["num_correct"]

    def report(self):
        return {
            "num_correct": torch.sum(self.num_correct),
            "per_class_score": self.num_correct,
            "repr": "Correct: %d" % torch.sum(self.num_correct)
        }

class Score(Metric):
    def __init__(self, num_classes, weight_classes=False, smooth=1e-6):
        super(Score, self).__init__()
        self.num_classes = num_classes
        self.weight_classes = weight_classes
        self.smooth = smooth

        self.numer = torch.zeros(self.num_classes)
        self.denom = torch.zeros(self.num_classes)
        self.updates_per_class = torch.zeros(self.num_classes)
        self.class_frequencies = torch.zeros(self.num_classes)

    @classmethod
    def from_config(cls, config):
        return cls(
            len(dict_get(config, "data", "labels", default=[])),
            dict_get(config, "params", "metrics_weight_classes", default=False),
            dict_get(config, "params", "smooth", default=1e-6)
        )
    
    @property
    def required_update_fields():
        return ["num_relevant", "num_predicted"]

    def reset(self):
        super(Score, self).reset()
        self.numer = torch.zeros(self.num_classes)
        self.denom = torch.zeros(self.num_classes)
        self.updates_per_class = torch.zeros(self.num_classes)
        self.class_frequencies = torch.zeros(self.num_classes)

    def get_values(self, update_state):
        return torch.zeros(self.num_classes), torch.zeros(self.num_classes)

    def update(self, update_state):
        super(Score, self).update(update_state)
        _numer, _denom = self.get_values(update_state)
        self.numer += _numer
        self.denom += _denom
        self.updates_per_class += ((update_state.get("num_relevant") + update_state.get("num_predicted")) > 0).float()
        self.class_frequencies += update_state.get("num_relevant")

    def calculate_score(self):
        per_class_score = self.numer / (self.denom + self.smooth)
        per_class_score[self.updates_per_class == 0] = float('nan')
        if self.weight_classes:
            class_weights = self.class_frequencies / torch.sum(self.class_frequencies)
            score = torch.nansum(per_class_score * class_weights)
        else:
            score = torch.nanmean(per_class_score)
        return score, per_class_score

    def report(self):
        score, per_class_score = self.calculate_score()
        return {
            "score": score,
            "per_class_score": per_class_score,
            "repr": "%s: %0.4f" % (type(self).__name__, score)
        }

class IoU(Score):
    @property
    def required_update_fields():
        return ["num_relevant", "num_predicted", "intersection"]
    
    def get_values(self, update_state):
        intersection = update_state.get("intersection")
        union = update_state.get("num_relevant") + update_state.get("num_predicted") - intersection
        return intersection, union

class Precision(Score):
    @property
    def required_update_fields():
        return ["num_relevant", "num_predicted", "intersection"]
    
    def get_values(self, update_state):
        return update_state.get("intersection"), update_state.get("num_predicted")

class Recall(Score):
    @property
    def required_update_fields():
        return ["num_relevant", "num_predicted", "intersection"]
    
    def get_values(self, update_state):
        return update_state.get("intersection"), update_state.get("num_relevant")

class Dice(Score):
    @property
    def required_update_fields():
        return ["num_relevant", "num_predicted", "intersection"]
    
    def get_values(self, update_state):
        intersection = update_state.get("intersection")
        denom = update_state.get("num_predicted") + update_state.get("num_relevant")
        return 2.0 * intersection, denom