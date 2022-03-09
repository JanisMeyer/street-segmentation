from time import time

from ...utils import dict_get

class Addon:
    @classmethod
    def from_config(cls, config):
        return cls()
    
    def reset(self):
        pass

    def update(self, update_state):
        pass

    def report(self):
        return {}

class Training(Addon):
    def __init__(self):
        self.start_time = None
        self.step = 0

    def reset(self):
        self.start_time = time()

    def update(self, update_state):
        self.step = update_state["step"]

    def report(self):
        return {
            "repr": "Finished training after %d steps and %0.2fs" % (self.step, time() - self.start_time)
        }

class TrainingProgress(Addon):
    def __init__(self, max_steps):
        self.max_steps = max_steps
        self.step = 0

    @classmethod
    def from_config(cls, config):
        return cls(dict_get(config, "train", "max_steps", default=0))

    def reset(self):
        self.step = 0

    def update(self, update_state):
        self.step += 1

    def report(self):
        return {
            "repr": "Step %{0}d/%d".format(len(str(self.max_steps))) % (self.step, self.max_steps)
        }

class Evaluation(Addon):
    def __init__(self):
        self.start_time = None
        self.num_samples = 0

    def reset(self):
        self.num_samples = 0
        self.start_time = time()

    def update(self, update_state):
        self.num_samples += update_state["targets"].size(0)

    def report(self):
        return {
            "repr": "Finished evaluation on %d samples in %0.2fs" % (self.num_samples, time() - self.start_time)
        }

class Inference(Addon):
    def __init__(self):
        self.start_time = None
        self.num_samples = 0

    def reset(self):
        self.num_samples = 0
        self.start_time = time()

    def update(self, update_state):
        self.num_samples += update_state["predictions"].size(0)

    def report(self):
        return {
            "repr": "Performed inference on %d samples in %0.2fs" % (self.num_samples, time() - self.start_time)
        }

class Timer(Addon):
    def __init__(self):
        self.start_time = None

    def reset(self):
        self.start_time = time()

    def report(self):
        return {
            "repr": "Elapsed: %0.2f" % (time() - self.start_time)
        }
    