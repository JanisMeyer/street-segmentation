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
        return {}

    def is_better(self, value):
        raise NotImplementedError

class Loss(Metric):
    def __init__(self):
        super(Loss, self).__init__()
        self.loss = 0.0
        self.losses = []

    def reset(self):
        super(Loss, self).reset()
        self.loss = 0.0
        self.losses = []

    def update(self, update_state):
        super(Loss, self).update(update_state)
        self.loss += update_state.get("loss", 0.0)
        if not self.losses:
            self.losses = [0] * len(update_state.get("losses", []))
            for i, loss in enumerate(update_state.get("losses", [])):
                self.losses[i] += loss

    def report(self):
        if len(self.losses) < 2:
            repr = ""
        else:
            repr = "(%s)" % " + ".join(["%0.4f" % (loss / self.num_updates) for loss in self.losses])
            
        return {
            "score": self.loss / self.num_updates,
            "repr": "Loss: %0.4f%s" % (self.loss / self.num_updates, repr)
        }

    def is_better(self, value):
        return (self.loss / self.num_updates) < value

class Accuracy(Metric):
    def __init__(self):
        super(Accuracy, self).__init__()
        self.num_correct = 0
        self.num_pixels = 0

    def reset(self):
        super(Accuracy, self).reset()
        self.num_correct = 0
        self.num_pixels = 0

    def update(self, update_state):
        super(Accuracy, self).update(update_state)
        self.num_correct += torch.sum(update_state.get("intersection"))
        self.num_pixels += torch.numel(update_state.get("targets"))

    def report(self):
        return {
            "score": self.num_correct / self.num_pixels,
            "repr": "Accuracy: %0.4f" % self.num_correct / self.num_pixels
        }

    def is_better(self, value):
        return (self.num_correct / self.num_pixels) > value

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

    def is_better(self, value):
        return self.calculate_score()[0] > value

class IoU(Score):
    def get_values(self, update_state):
        intersection = update_state.get("intersection")
        union = update_state.get("num_relevant") + update_state.get("num_predicted") - intersection
        return intersection, union

class Precision(Score):
    def get_values(self, update_state):
        return update_state.get("intersection"), update_state.get("num_predicted")

class Recall(Score):
    def get_values(self, update_state):
        return update_state.get("intersection"), update_state.get("num_relevant")

class Dice(Score):
    def get_values(self, update_state):
        intersection = update_state.get("intersection")
        denom = update_state.get("num_predicted") + update_state.get("num_relevant")
        return 2.0 * intersection, denom