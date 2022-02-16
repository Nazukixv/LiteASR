"""Event trigger during training."""


class Trigger(object):

    def __init__(self, interval: int, unit: str):
        assert unit in ["epoch", "iteration"]
        self.interval = interval
        self.unit = unit
        self.prev_unit = 0

    def is_triggered(self, trainer):
        if self.unit == "epoch":
            if trainer.epoch == self.prev_unit + self.interval:
                self.prev_unit += self.interval
                return True
            else:
                return False
        else:
            if trainer.iter == self.prev_unit + self.interval:
                self.prev_unit += self.interval
                return True
            else:
                return False
