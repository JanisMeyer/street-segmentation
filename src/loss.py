import torch
import torch.nn.functional as F

class Loss(torch.nn.Module):
    def __init__(self, class_weight_fn=None, smooth=1e-6):
        super(Loss, self).__init__()
        self.class_weights = None
        self.class_weight_fn = class_weight_fn
        self.smooth = smooth

    def load_class_weights(self, device, class_frequencies):
        if self.class_weight_fn is None:
            return
        self.class_weights = self.class_weight_fn(class_frequencies).to(device)

    @staticmethod
    def from_config(loss_config):
        if isinstance(loss_config, list):
            return CompositeLoss.from_config(loss_config)
        else:
            loss_fn = LOSSES[loss_config.get("criterion", "").lower()]
            loss_fn = loss_fn.from_config(loss_config)
            return loss_fn

class CompositeLoss(Loss):
    def __init__(self, criterions, weights=None):
        super(Loss, self).__init__()
        self.criterions = torch.nn.ModuleList(criterions)
        self.weights = weights if weights else [1.0] * len(self.criterions)

    @staticmethod
    def from_config(loss_configs):
        criterions = []
        weights = []
        for loss_config in loss_configs:
            criterions.append(Loss.from_config(loss_config))
            weights.append(loss_config.get("weight", 1.0))
        return CompositeLoss(criterions, weights)

    def load_class_weights(self, device, class_frequencies):
        for criterion in self.criterions:
            criterion.load_class_weights(device, class_frequencies)

    def forward(self, inputs, targets):
        losses = []
        for weight, criterion in zip(self.weights, self.criterions):
            losses += [weight * criterion(inputs, targets)]
        return {
            "loss": sum(losses),
            "losses": losses
        }
        
class CrossEntropyLoss(Loss):
    @staticmethod
    def from_config(loss_config):
        return CrossEntropyLoss(loss_config.get("gamma", 2.0),
                                class_weight_fn=CLASS_WEIGHT_FNS.get(loss_config.get("class_weights", "").lower(), None))

    def forward(self, inputs, targets):
        return {
            "loss": F.cross_entropy(inputs["logits"], targets, weight=self.class_weights)
        }

class FocalLoss(Loss):
    def __init__(self, gamma, class_weight_fn=None, smooth=1e-6):
        super(Loss, self).__init__(class_weight_fn=class_weight_fn, smooth=smooth)
        self.gamma = gamma

    @staticmethod
    def from_config(loss_config):
        return FocalLoss(loss_config.get("gamma", 2.0),
                         class_weight_fn=CLASS_WEIGHT_FNS.get(loss_config.get("class_weights", "").lower(), None))
    
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs["logits"], targets, weight=self.class_weights, reduction='none')
        pt = torch.gather(inputs["probabilities"], dim=1, index=targets[:, None, ...])
        loss = torch.pow(1 - pt, self.gamma) * ce_loss
        return {
            "loss": torch.mean(loss)
        }

class DiceLoss(Loss):
    def forward(self, inputs, targets):
        targets_one_hot = F.one_hot(targets, num_classes=inputs["probabilities"].size(1)).permute(0, 3, 1, 2)

        intersection = torch.sum(inputs["probabilities"] * targets_one_hot, dim=(0, 2, 3))
        denom = torch.sum(inputs["probabilities"] + targets_one_hot, dim=(0, 2, 3))
        loss = 1.0 - torch.mean((2.0 * intersection + self.smooth) / (denom + self.smooth))
        return {
            "loss": loss
        }

    @staticmethod
    def from_config(loss_config):
        return DiceLoss()

def inverse_class_frequency(class_frequencies):
    weights = torch.reciprocal(class_frequencies)
    weights[weights == float("Inf")] = 0.0
    weights = weights / weights.sum()
    return weights

def inverse_squared_class_frequency(class_freqiencies):
    weights = torch.reciprocal(torch.square(class_freqiencies))
    weights[weights == float("Inf")] = 0.0
    weights = weights / weights.sum()
    return weights

CLASS_WEIGHT_FNS = {
    "inv_freq": inverse_class_frequency,
    "inv_sq_freq": inverse_squared_class_frequency
}

LOSSES = {
    "ce": CrossEntropyLoss,
    "focal": FocalLoss,
    "dice": DiceLoss
}