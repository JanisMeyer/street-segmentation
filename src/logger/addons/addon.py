from time import time

from ...utils import dict_get

class Addon:
    @classmethod
    def from_config(cls, config):
        return cls()

    @property
    def required_update_fields():
        return []

    def reset(self):
        pass

    def update(self, update_state):
        pass

    def report(self):
        return {}

    def load_state_dict(self, state_dict):
        return

    def state_dict(self):
        return {}

class Training(Addon):
    def __init__(self):
        self.timer = None
        self.current_step = 0
        self.base_time = 0

    def reset(self):
        self.timer = time()

    def update(self, update_state):
        self.current_step += update_state["num_steps"]

    def report(self):
        elapsed = time() - self.timer + self.base_time
        return {
            "current_step": self.current_step,
            "elapsed": elapsed,
            "repr": "Finished training after %d steps and %0.2fs" % (self.current_step, elapsed)
        }

    def load_state_dict(self, state_dict):
        if "current_step" in state_dict:
            self.current_step = state_dict["current_step"]
        if "base_time" in state_dict:
            self.base_time = state_dict["base_time"]

    def state_dict(self):
        return {
            "current_step": self.current_step,
            "base_time": time() - self.timer + self.base_time
        }

class TrainingProgress(Addon):
    def __init__(self, max_steps):
        self.max_steps = max_steps
        self.current_step = 0

    @classmethod
    def from_config(cls, config):
        return cls(dict_get(config, "train", "max_steps", default=0))

    def reset(self):
        self.current_step = 0

    def update(self, update_state):
        self.current_step += 1

    def report(self):
        return {
            "current_step": self.current_step,
            "repr": "Step %{0}d/%d".format(len(str(self.max_steps))) % (self.current_step, self.max_steps)
        }

    def load_state_dict(self, state_dict):
        if "current_step" in state_dict:
            self.max_steps += state_dict["current_step"]
            self.current_step = state_dict["current_step"]

class Evaluation(Addon):
    def __init__(self):
        self.timer = None
        self.num_samples = 0

    @property
    def required_update_fields():
        return ["num_samples"]

    def reset(self):
        self.num_samples = 0
        self.timer = time()

    def update(self, update_state):
        self.num_samples += update_state["num_samples"]

    def report(self):
        elapsed = time() - self.timer
        return {
            "num_samples": self.num_samples,
            "elapsed": elapsed,
            "repr": "Finished evaluation on %d samples in %0.2fs" % (self.num_samples, elapsed)
        }

class Inference(Addon):
    def __init__(self):
        self.timer = None
        self.num_samples = 0

    @property
    def required_update_fields():
        return ["num_samples"]

    def reset(self):
        self.num_samples = 0
        self.timer = time()

    def update(self, update_state):
        self.num_samples += update_state["num_samples"]

    def report(self):
        elapsed = time() - self.timer
        return {
            "num_samples": self.num_samples,
            "elapsed": elapsed,
            "repr": "Performed inference on %d samples in %0.2fs" % (self.num_samples, elapsed)
        }

class Timer(Addon):
    def __init__(self):
        self.timer = None

    def reset(self):
        self.timer = time()

    def report(self):
        elapsed = time() - self.start_time
        return {
            "elapsed": elapsed,
            "repr": "Elapsed: %0.2f" % elapsed
        }
    