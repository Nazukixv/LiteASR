"""Event trigger during training."""

from functools import wraps


class Trigger(object):
    """Event trigger object.

    :param int interval: time interval of the triggered event
    :param str unit: interval unit, either `epoch` or `iteration`

    """

    def __init__(self, interval: int, unit: str):
        assert unit in ["epoch", "iteration"]
        self.interval = interval
        self.unit = unit
        self.prev_unit = 0

    def is_triggered(self, trainer, unit: str) -> bool:
        criter = trainer.epoch if unit == "epoch" else trainer.iter
        if unit == self.unit and criter == self.prev_unit + self.interval:
            self.prev_unit += self.interval
            return True
        else:
            return False

    def __call__(self, event):

        @wraps(event)
        def wrapper(trainer, unit):
            if self.is_triggered(trainer, unit):
                event(trainer)

        return wrapper


class EventManager(object):
    """Manage the triggered event

    :Example:

    >>> # inside `Trainer`
    >>> self.event_manager = EventManager()
    >>> self.event_manager.add_event(some_event)
    >>> self.event_manager.trigger_events("epoch")

    >>> @Trigger(2, "epoch")
    >>> def some_event(self):
    >>>     ...

    `some_event` will therefore run each 2 epoch.

    """

    def __init__(self):
        self.events = []

    def add_event(self, event):
        self.events.append(event)

    def trigger_events(self, unit):
        for event in self.events:
            event(unit)
