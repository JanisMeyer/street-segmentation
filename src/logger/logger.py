import sys
import logging

import torch

logger = None

def init_logger():
    global logger
    logger = Logger()
    logger.log_info("Initialized logger...")

def log_info(message, *args):
    global logger
    if not logger:
        init_logger()
    logger.log_info(message, *args)

def log_per_class(statistics, id_to_label, sort_by=None):
    global logger
    if not logger:
        init_logger()
    logger.log_per_class(statistics, id_to_label, sort_by=sort_by)

class Logger:
    def __init__(self):
        logging.getLogger().handlers.clear()

        stdout_handler = logging.StreamHandler(stream=sys.stdout)
        stdout_handler.setLevel(getattr(logging, 'INFO'))
        stdout_handler.setFormatter(logging.Formatter(fmt="%(asctime)s: %(message)s", datefmt="%H:%M:%S"))
        logging.getLogger().addHandler(stdout_handler)

        logging.getLogger().setLevel(getattr(logging, 'INFO'))

    def log_info(self, message, *args):
        logging.getLogger().info(message, *args)

    def log_per_class(self, statistics, id_to_label, sort_by=None):
        num_classes = list(statistics.values())[0].size(0)
        output = "\n%-20.20s\t%s" % ("class label", "  ".join(["%9.12s" % key for key in statistics]))
        
        indices = list(range(num_classes))
        if sort_by is not None:
            indices = sorted(indices, key=lambda x: statistics[sort_by][x], reverse=True)    
        
        for class_idx in indices:
            scores = [stats[class_idx] for stats in statistics.values()]
            output += "\n%-20.20s\t%s" % (id_to_label(class_idx),
                                          "  ".join(["%9.4f" % score if not score.dtype == torch.int else "%9d" % score for score in scores]))
        scores = [torch.nanmean(stats) if not stats.dtype == torch.int else torch.nansum(stats, dtype=torch.int) for stats in statistics.values()]
        output += "\n%-20.20s\t%s" % ("", "  ".join(["%9.4f" % score if not score.dtype == torch.int else "%9d" % score for score in scores]))
        self.log_info(output)
