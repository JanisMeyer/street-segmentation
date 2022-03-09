import os
from time import time

import torch
import torch.nn.functional as F

from .logger import log_info, log_per_class
from ..utils import dict_get, dict_set

class Reporter:
    def __init__(self, id_to_label=None, metric_for_best=None, key_for_best="eval"):
        self.id_to_label = id_to_label

        self.addons = {}
        self.histograms = {}

        self.metric_for_best = metric_for_best
        self.key_for_best = key_for_best
        self.best_value = None
        self.best_step = 0

    def add_addons(self, key, addons):
        if key not in self.addons:
            self.addons[key] = addons
        else:
            self.addons[key] += addons

    def reset(self, key):
        for addon in self.addons[key]:
            addon.reset()
        if self.metric_for_best and key == self.key_for_best:
            self.metric_for_best.reset()

    def update(self, key, update_state):
        for addon in self.addons[key]:
            addon.update(update_state)
        if self.metric_for_best and key == self.key_for_best:
            self.metric_for_best.update(update_state)

    def report(self, key, log_results=True, log_class_results=False):
        report = {}
        for addon in self.addons[key]:
            name = type(addon).__name__
            report[name] = addon.report()
            if "score" in report[name]:
                if name not in self.histograms:
                    self.histograms[name] = {key: []}
                if key not in self.histograms[name]:
                    self.histograms = dict_set(self.histograms, [], name, key)
                self.histograms[name][key].append(dict_get(report, name, "score"))
        
        if log_results:
            log_info(";".join([dict_get(report, key, "repr", default="") for key in report if dict_get(report, key, "repr", default="")]))
        
        if log_class_results and self.id_to_label is not None:
            per_class_scores = {
                key: dict_get(report, key, "per_class_score") for key in report if "per_class_score" in report[key]
            }
            log_per_class(per_class_scores, self.id_to_label)

        if self.metric_for_best and key == self.key_for_best:
            if self.best_value is None or self.metric_for_best.is_better(self.best_value):
                 self.best_value = self.metric_for_best.report()["score"]
                 self.best_step = 
        return report